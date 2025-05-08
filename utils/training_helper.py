class TrainingSchedule:
    def __init__(self, max_nenvs, max_nruns, max_warmup_episodes, max_nupdates):
        self.nenvs_stage_1 = max(int(max_nenvs//2.5), 1)
        self.nenvs_stage_2 = max(max_nenvs, 1)
        
        self.nruns_stage_1 = max(max_nruns//2, 1)
        self.nruns_stage_2 = max(max_nruns, 1)
        
        self.nupdates_stage_1 = max(max_nupdates//200, 1)
        self.nupdates_stage_2 = max(max_nupdates//50, 1)
        self.nupdates_stage_3 = max(max_nupdates//5, 1)
        
        self.warmup_stage_1 = max(max_warmup_episodes//5, 1)
        self.warmup_stage_2 = max(max_warmup_episodes, 1)

    def training_schedule(self, n_episode):
        """Training Schedule During Warmup

        Args:
            n_episode: current episode number

        Returns:
            n_threads: number of parallel threads to spawm
            n_runs: number of runs to perform per thread
            n_updates: number of updates to perform per thread
        """
        if n_episode < self.warmup_stage_1:
            return self.nenvs_stage_1, self.nruns_stage_1, self.nupdates_stage_1
        elif n_episode < self.warmup_stage_2:
            return self.nenvs_stage_1, self.nruns_stage_2, self.nupdates_stage_2
        else:
            return self.nenvs_stage_2, self.nruns_stage_2, self.nupdates_stage_3