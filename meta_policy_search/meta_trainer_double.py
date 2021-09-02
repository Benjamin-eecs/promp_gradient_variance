import tensorflow as tf
import numpy as np
import time
from meta_policy_search.utils import logger
from scipy.spatial.distance import cdist


class Trainer(object):
    """
    Performs steps of meta-policy search.

     Pseudocode::

            for iter in n_iter:
                sample tasks
                for task in tasks:
                    for adapt_step in num_inner_grad_steps
                        sample trajectories with policy
                        perform update/adaptation step
                    sample trajectories with post-update policy
                perform meta-policy gradient step(s)

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            algo,
            env,
            sample_processor,
            policy,
            n_itr,
            start_itr=0,
            num_inner_grad_steps=1,
            sess=None,
            pg_sampler=None,
            h_sampler=None,
            outer_sampler=None,
            num_sapling_rounds = 10

            ):
        self.algo = algo
        self.env = env
        self.pg_sampler = pg_sampler
        assert pg_sampler is not None
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_inner_grad_steps = num_inner_grad_steps

        self.h_sampler          = h_sampler
        self.outer_sampler      = outer_sampler
        self.num_sapling_rounds = num_sapling_rounds

        if sess is None:
            sess = tf.Session()
        self.sess = sess



    def train(self):
        """
        Trains policy on env using algo

        Pseudocode::
        
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """
        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")

                gradients = []


                for i in range(self.num_sapling_rounds):
                    logger.log("\n ----- Sampling Round %d ---" % i)


                    dry = i < self.num_sapling_rounds-1
                    
                    if i==0:
                        tasks = self.pg_sampler.update_tasks()
                    self.pg_sampler.set_tasks(tasks)
                    self.h_sampler.set_tasks(tasks)
                    self.outer_sampler.set_tasks(tasks)
                    self.policy.switch_to_pre_update()  # Switch to pre-update policy

                    all_samples_data, all_paths = [], []
                    list_sampling_time, list_inner_step_time, list_outer_step_time, list_proc_samples_time = [], [], [], []
                    start_total_inner_time = time.time()
                    #original update
                    for step in range(self.num_inner_grad_steps+1):
                        logger.log('** Step ' + str(step) + ' **')

                        """ -------------------- Sampling --------------------------"""

                        logger.log("Obtaining samples...")
                        time_env_sampling_start = time.time()
                        if step == 0:
                            paths_pg = self.pg_sampler.obtain_samples(log=True, log_prefix='Step_%d-' % step)
                            paths_h  = self.h_sampler.obtain_samples()
                        elif step == 1:
                            paths_outer = self.outer_sampler.obtain_samples(log=True, log_prefix='Step_%d-' % step)
                        list_sampling_time.append(time.time() - time_env_sampling_start)
                        #all_paths.append(paths)

                        """ ----------------- Processing Samples ---------------------"""

                        logger.log("Processing samples...")
                        time_proc_samples_start = time.time()
                        if step == 0:
                            samples_data_pg = self.sample_processor.process_samples(paths_pg, log='all', log_prefix='Step_%d-' % step)
                            samples_data_h = self.sample_processor.process_samples(paths_h)#?
                            all_samples_data.append(samples_data_pg)
                        else:
                            samples_outer = self.sample_processor.process_samples(paths_outer, log='all', log_prefix='Step_%d-' % step)
                            all_samples_data.append(samples_outer)
                        list_proc_samples_time.append(time.time() - time_proc_samples_start)



                        """ ------------------- Inner Policy Update --------------------"""

                        time_inner_step_start = time.time()
                        if step < self.num_inner_grad_steps:
                            logger.log("Computing inner policy updates...")
                            self.algo._adapt(samples_data_pg)
                        list_inner_step_time.append(time.time() - time_inner_step_start)


                    all_samples_data.append(samples_data_h)

                    """ compute gradients """
                    gradients.append(self.algo.compute_gradients(all_samples_data))

                    if not dry:
                        """ ------------ Compute and log gradient variance ------------"""
                        # compute variance of adaptation gradients
                        for step_id in range(self.num_inner_grad_steps):
                            meta_batch_size = len(gradients[0][0])
                            grad_std, grad_rstd = [], []
                            for task_id in range(meta_batch_size):
                                stacked_grads = np.stack([gradients[round_id][step_id][task_id]
                                                          for round_id in range(self.num_sapling_rounds)], axis=1)
                                std = np.std(stacked_grads, axis=1)
                                mean = np.abs(np.mean(stacked_grads, axis=1))
                                grad_std.append(np.mean(std))
                                grad_rstd.append(np.mean(std/mean))

                            logger.logkv('Step_%i-GradientMean', np.mean(mean))
                            logger.logkv('Step_%i-GradientStd'%step_id, np.mean(grad_std))
                            logger.logkv('Step_%i-GradientRStd' % step_id, np.mean(grad_rstd))

                        # compute variance of meta gradients
                        stacked_grads = np.stack([gradients[round_id][self.num_inner_grad_steps]
                                                  for round_id in range(self.num_sapling_rounds)], axis=1)
                        std = np.std(stacked_grads, axis=1)
                        mean = np.abs(np.mean(stacked_grads, axis=1))

                        meta_grad_std = np.mean(std)
                        meta_grad_rstd = np.mean(std/(mean + 1e-8))
                        meta_grad_rvar = np.mean(std**2/(mean + 1e-8))

                        logger.logkv('Meta-GradientMean', np.mean(mean))
                        logger.logkv('Meta-GradientStd', meta_grad_std)
                        logger.logkv('Meta-GradientRStd', meta_grad_rstd)
                        logger.logkv('Meta-GradientRVariance', meta_grad_rvar)

                        # compute cosine dists
                        cosine_dists = cdist(np.transpose(stacked_grads), np.transpose(np.mean(stacked_grads, axis=1).reshape((-1, 1))),
                                             metric='cosine')
                        mean_abs_cos_dist = np.mean(np.abs(cosine_dists))
                        mean_squared_cosine_dists = np.mean(cosine_dists**2)
                        mean_squared_cosine_dists_sqrt = np.sqrt(mean_squared_cosine_dists)

                        logger.logkv('Meta-GradientCosAbs', mean_abs_cos_dist)
                        logger.logkv('Meta-GradientCosVar', mean_squared_cosine_dists)
                        logger.logkv('Meta-GradientCosStd', mean_squared_cosine_dists_sqrt)




                        total_inner_time = time.time() - start_total_inner_time                
                        time_maml_opt_start = time.time()
                        """ ------------------ Outer Policy Update ---------------------"""

                        logger.log("Optimizing policy...")
                        # This needs to take all samples_data so that it can construct graph for meta-optimization.
                        time_outer_step_start = time.time()
                        self.algo.optimize_policy(all_samples_data)

                        """ ------------------- Logging Stuff --------------------------"""
                        logger.logkv('Itr', itr)
                        logger.logkv('n_timesteps', self.pg_sampler.total_timesteps_sampled+self.outer_sampler.total_timesteps_sampled)

                        logger.logkv('Time-OuterStep', time.time() - time_outer_step_start)
                        logger.logkv('Time-TotalInner', total_inner_time)
                        logger.logkv('Time-InnerStep', np.sum(list_inner_step_time))
                        logger.logkv('Time-SampleProc', np.sum(list_proc_samples_time))
                        logger.logkv('Time-Sampling', np.sum(list_sampling_time))

                        logger.logkv('Time', time.time() - start_time)
                        logger.logkv('ItrTime', time.time() - itr_start_time)
                        logger.logkv('Time-MAMLSteps', time.time() - time_maml_opt_start)


                        logger.log("Saving snapshot...")
                        params = self.get_itr_snapshot(itr)
                        logger.save_itr_params(itr, params)
                        logger.log("Saved")
                logger.dumpkvs()



        logger.log("Training finished")
        self.sess.close()        

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
