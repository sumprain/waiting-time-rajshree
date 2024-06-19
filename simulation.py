import numpy as np
import numpy.typing as npt
import scipy.stats as st #type: ignore
import matplotlib.pyplot as plt #type:ignore
import statistics as sts

from queue import Queue, Empty
from collections import namedtuple, defaultdict, OrderedDict

av_waiting_time = namedtuple("av_waiting_time", "mean cl_95".split())
max_queue_length = namedtuple("max_queue_length", "max cl_95".split())
desc_num_entities = namedtuple("desc_num_entities", "mean cl_95 median max min".split())

class FitError (Exception):
    pass

class Generator:
    
    def __init__ (self, dist_func: st.rv_discrete) -> None:
        '''
            dist_func: scipy stats distribution function
        '''
        self.dist: st.rv_discrete = dist_func
        self.sample_log: list[int | float] = []
        self.no_work_ind: list[int] = []
        self.ts_sample_log: list[int] = []
        self.ts_no_work_ind: list[int] = []

    def set_params (self, **kwargs) -> None:
        '''
            kwargs: param = value
                e.g., for poisson dist: mu = 10
        '''
        self.dist_init: st.rv_discrete = self.dist(**kwargs)

    def sample (self, t: int, n_samp: int = 1) -> npt.NDArray[np.int32]:
        samp = self.dist_init.rvs(size = n_samp)
        self.sample_log.append(samp)
        self.ts_sample_log.append(t)
        return samp

    def no_work (self, t: int) -> None:
        self.no_work_ind.append(1)
        self.ts_no_work_ind.append(t)

    def yes_work (self, t: int) -> None:
        self.no_work_ind.append(0)
        self.ts_no_work_ind.append(t)

    def get_params_from_data (self, data: npt.NDArray[np.int32], \
        bounds: dict[str, tuple[float | int]], ax = None, plot = False):
        
        fit = st.fit(self.dist, data = data, bounds = bounds)

        if plot:
            if ax is None:
                f, ax = plt.subplots()
                fit.plot(ax)
            else:
                fit.plot(ax = ax)        
            plt.show()
            
        if not fit.success:
            raise FitError()
        else:
            return fit.params

class Entity:

    def __init__ (self, st_time: int) -> None:
        self.st_time: int = st_time

    def set_end_time (self, end_time: int) -> None:
        if end_time < self.st_time:
            raise ValueError("end time cannot be smaller that start time")

        self.end_time: int = end_time

    def waiting_time(self) -> int:
        return self.end_time - self.st_time

class Simulator:

    def __init__ (self, arr_dist: st.rv_discrete, exit_dists: tuple[st.rv_discrete]) -> None:
        
        self.arr_gen: Generator = Generator(dist_func = arr_dist)
        
        self.exit_gens: list[Generator] = []
        
        for dist in exit_dists:
            self.exit_gens.append(Generator(dist))

        self.queue: Queue = Queue(maxsize = -1)
        self.queue_len: list[int] = []
        self.ts_queue_len: list[int] = []

        self.entities: list[Entity] = []

    def reset (self) -> None:
        self.queue_len = []
        self.ts_queue_len = []
        self.entities = []
        self.arr_gen.no_work_ind = []
        self.arr_gen.sample_log = []
        self.arr_gen.ts_no_work_ind = []
        self.arr_gen.ts_sample_log = []
        
        for exit_gen in self.exit_gens:
            exit_gen.no_work_ind = []
            exit_gen.sample_log = []
            exit_gen.ts_no_work_ind = []
            exit_gen.ts_sample_log = []

    def simulate(self, arr_params: dict[int, dict], exit_params: dict[int, tuple[dict]], end_time: int) -> None:

        '''
            *_params: dict[time_start_of_params, named params] of *
        '''
        
        self.end_time: int = end_time

        arr_times_at_change = list(arr_params.keys())
        exit_times_at_change = list(exit_params.keys())

        arr_times_at_change.sort()
        exit_times_at_change.sort()

        for t in range(self.end_time):

            if t in arr_times_at_change:
                self.arr_gen.set_params(**arr_params[t])

            arrs = self.arr_gen.sample(t, 1)[0]
            for _ in range(arrs):
                self.queue.put_nowait(Entity(st_time = t))
            
            if t in exit_times_at_change:
                for i, param in enumerate(exit_params[t]):
                    self.exit_gens[i].set_params(**param)

            for exit_gen in self.exit_gens:
                exits = exit_gen.sample(t, 1)[0]
                for _ in range(exits):
                    try:
                        exited_entity = self.queue.get_nowait()
                        exited_entity.set_end_time(t)
                        self.entities.append(exited_entity)
                        exit_gen.yes_work(t)
                    except Empty:
                        exit_gen.no_work(t)

            self.queue_len.append(self.queue.qsize())
            self.ts_queue_len.append(t)

        t = self.end_time + 1
        
        # for disposing off entities remaining after end time.
        while self.queue.qsize() > 0:
            if t in exit_times_at_change:
                for i, param in enumerate(exit_params[t]):
                    self.exit_gens[i].set_params(**param)

            for exit_gen in self.exit_gens:
                exits = exit_gen.sample(t, 1)[0]
                for _ in range(exits):
                    try:
                        exited_entity = self.queue.get_nowait()
                        exited_entity.set_end_time(t)
                        self.entities.append(exited_entity)
                        exit_gen.yes_work(t)
                    except Empty:
                        exit_gen.no_work(t)
            
            self.queue_len.append(self.queue.qsize())
            self.ts_queue_len.append(t)

            t = t + 1

    
class MasterSimulator:
    
    def __init__ (self, arr_dist, exit_dists, n_sim = 1000):
        self.n_sim = n_sim
        self.simulations = []
        for i in range(self.n_sim):
            self.simulations.append(Simulator(arr_dist, exit_dists))
        self.start_times_for_waiting_times = []
        self.waiting_times = defaultdict(list)
        self.queue_lengths = defaultdict(list)

    def simulate (self, arr_params: dict[int, dict], exit_params: dict[int, tuple[dict]], end_time: int):
        for sim in self.simulations:
            sim.simulate(arr_params, exit_params, end_time)

    def reset (self):
        for sim in self.simulations:
            sim.reset()

    def get_num_entities (self):

        num_entities = []
        for sim in self.simulations:
            num_entities.append(len(sim.entities))
        
        return num_entities

    def describe_num_entities (self):
        
        num_entities = self.get_num_entities()

        return desc_num_entities(mean = np.mean(num_entities), cl_95 = st.bootstrap((num_entities, ), np.mean).confidence_interval, \
            median = sts.median(num_entities), max = max(num_entities), min = min(num_entities))

    def get_waiting_times (self):
    
        for sim in self.simulations:
            for ent in sim.entities:
                self.waiting_times[ent.st_time].append(ent.waiting_time())
        
        sorted_ts = sorted(list(self.waiting_times.keys()))

        wt_ts = {}

        for t in sorted_ts:
            wt_ts[t] = self.waiting_times[t]
        
        self.waiting_times = wt_ts

    def average_waiting_time (self, with_time = False, with_ci = False):
        
        self.get_waiting_times()

        if not with_time:
            wt_ts = []
            for t in self.waiting_times:
                wt_ts.append(self.waiting_times[t])
                wt_ts = [i for l in wt_ts for i in l]
                if with_ci:
                    return av_waiting_time(mean = sts.mean(wt_ts), cl_95 = st.bootstrap((wt_ts, ), np.mean).confidence_interval)
                else:
                    return av_waiting_time(mean = sts.mean(wt_ts), cl_95 = None)
        else:
            wt_ts = {}
            for t in self.waiting_times:
                if with_ci:
                    wt_ts[t] = av_waiting_time(mean = sts.mean(self.waiting_times[t]), \
                        cl_95 = st.bootstrap((self.waiting_times[t], ), np.mean).confidence_interval)
                else:
                    wt_ts[t] = av_waiting_time(mean = sts.mean(self.waiting_times[t]), cl_95 = None)

            return wt_ts

    def plot_average_waiting_time_with_time (self, ax, with_ci = False):
        wt_ts = self.average_waiting_time(with_time = True, with_ci = with_ci)
        ax.plot(wt_ts.keys(), [wt_ts[t].mean for t in wt_ts], 'r-')
        if with_ci:
            ax.plot(wt_ts.keys(), [wt_ts[t].cl_95[0] for t in wt_ts], 'b--')
            ax.plot(wt_ts.keys(), [wt_ts[t].cl_95[1] for t in wt_ts], 'b--')

    def get_queue_lengths (self):
        for sim in self.simulations:
            for i in range(len(sim.ts_queue_len)):
                self.queue_lengths[sim.ts_queue_len[i]].append(sim.queue_len[i])
    
    def max_queue_length (self, with_time = False, with_ci = False):
        
        self.get_queue_lengths()
        
        if not with_time:
            ql_ts = []
            for t in self.queue_lengths:
                ql_ts.append(self.queue_lengths[t])
            
            ql_ts = [i for l in ql_ts for i in l]    
            if with_ci:
                return max_queue_length(max = max(ql_ts), cl_95 = st.bootstrap((ql_ts, ), np.max, method = 'basic').confidence_interval)
            else:
                return max_queue_length(max = max(ql_ts), cl_95 = None)

        else:
            ql_ts = {}
            for t in self.queue_lengths:
                if with_ci:
                    ql_ts[t] = max_queue_length(max = max(self.queue_lengths[t]), \
                        cl_95 = st.bootstrap((self.queue_lengths[t], ), np.max).confidence_interval)
                else:
                    ql_ts[t] = max_queue_length(max = max(self.queue_lengths[t]), cl_95 = None)

            return ql_ts 
    
    def plot_max_queuelength_with_time (self, ax, with_ci = False):
        ql_ts = self.max_queue_length(with_time = True, with_ci = with_ci)
        ax.plot(ql_ts.keys(), [ql_ts[t].max for t in ql_ts], 'r-')
        if with_ci:
            ax.plot(ql_ts.keys(), [ql_ts[t].cl_95[0] for t in ql_ts], 'b--')
            ax.plot(ql_ts.keys(), [ql_ts[t].cl_95[1] for t in ql_ts], 'b--')

def create_master_simulator (arrivals, exits, durations, n_exits, thresh_rate = 0.05, n_sim = 500):
    
    arr_rates = [a / d for a, d in zip(arrivals, durations)]
    ars = []
    for ar in arr_rates:
        if ar < thresh_rate:
            ars.append(thresh_rate)
        else:
            ars.append(ar)
    arr_rates = ars

    exit_rates = [e / (d * n_exits) for e, d in zip(exits, durations)]
    ers = []
    for er in exit_rates:
        if er < thresh_rate:
            ers.append(thresh_rate)
        else:
            ers.append(er)
    exit_rates = ers
    
    m_simulator = MasterSimulator(st.poisson, [st.poisson] * n_exits, n_sim)

    arr_params = {}
    pres_arr_time = 0
    for i in range(len(arr_rates)):
        arr_params[pres_arr_time] = {'mu': arr_rates[i]}
        pres_arr_time += durations[i]
    
    exit_params = {}
    pres_exit_time = 0
    for i in range(len(exit_rates)):
        exit_params[pres_exit_time] = [{'mu': exit_rates[i]}] * n_exits
        pres_exit_time += durations[i]
    
    m_simulator.reset()
    m_simulator.simulate(arr_params, exit_params, end_time = pres_arr_time) 

    return m_simulator