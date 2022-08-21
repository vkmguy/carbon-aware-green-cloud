from multiprocessing import Pool
from os import makedirs
from typing import List, Iterator, Tuple

import numpy as np
import pandas as pd
import simpy
import logging

from datetime import timedelta
from leaf.infrastructure import Infrastructure
from leaf.power import PowerMeter
from infrastructure import Cloud
from settings import MEASUREMENT_INTERVAL, COUNTRIES, ERROR_REPETITIONS
from strategy import Strategy, AdHocStrategy

Job = Tuple[int, int, int]  # id, arrival time, duration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main(node, ci, jobs: List[Job], strategy: Strategy):
    env = simpy.Environment()
    power_meter = PowerMeter(node, measurement_interval=MEASUREMENT_INTERVAL)
    env.process(datacenter_process(env, jobs, strategy))
    env.process(power_meter.run(env))
    # logger.info(f"Starting simulation with strategy {strategy}...")
    env.run(until=len(ci) * MEASUREMENT_INTERVAL)
    # Watt usage at point in time
    return np.array([float(measurement) for measurement in power_meter.measurements])


def _print_analysis(consumption_timeline, ci):
    consumed_kwh = sum(consumption_timeline) * MEASUREMENT_INTERVAL / 60 / 1000

    # gCO2/timestep = gCO2/kWh * W / 1000 / steps_per_hour
    gco2_per_timestep = np.multiply(ci.values, consumption_timeline) * MEASUREMENT_INTERVAL / 60 / 1000
    emitted_gco2 = sum(gco2_per_timestep)

    logger.info(f"- Consumed {consumed_kwh:.2f} kWh and emitted {emitted_gco2 / 1000:.2f} kg CO2e.")
    logger.info(f"- Average CO2 intensity of used energy was {emitted_gco2 / consumed_kwh:.2f} gCO2e/kWh.")


def datacenter_process(env: simpy.Environment, jobs: Iterator[Job], strategy: Strategy):
    for _, arrival_time, duration in jobs:
        time_until_arrival = arrival_time - env.now
        if time_until_arrival < 0:
            continue
        yield env.timeout(time_until_arrival)
        env.process(strategy.run(env, duration))


def adhoc_worker(node, jobs, ci, error, seed, forecast_method, interruptible):
    strategy = AdHocStrategy(node=node, ci=_apply_error(ci, error, seed),
                             interval=MEASUREMENT_INTERVAL, forecast=forecast_method, interruptible=interruptible)
    return main(node, ci, jobs, strategy)


def ml_experiment():
    ml_jobs = generate_batch_jobs(n=3387,
                                  min_duration=4 * 60,
                                  max_duration=90 * 60,
                                  workday_start=9 * 60,
                                  workday_end=17 * 60,
                                  workdays_only=True)
    for country in COUNTRIES:
        # Load CI data
        ci = _load_dataset(f"data/{country}_ci.csv")
        ci = ci[ci.index.minute % MEASUREMENT_INTERVAL == 0]

        # Baseline Experiment
        ad_hoc_experiment("ml", country, ml_jobs, ci, forecast_method=0, interruptible=False, error=False)

        # Experiments
        for interruptible in [True, False]:
            for error in [0, 0.05, 0.1]:
                for forecast_method in ["next_workday", "semi_weekly", "weekend"]:
                    ad_hoc_experiment("ml", country, ml_jobs, ci,
                                      forecast_method=forecast_method,
                                      interruptible=interruptible,
                                      error=error)


def ad_hoc_experiment(experiment_name: str,
                      country: str,
                      jobs: List,
                      ci,
                      forecast_method,
                      interruptible: bool,
                      error: float):
    logger.info(f"ad_hoc_experiment({experiment_name}, forecast_method={forecast_method}, "
          f"interruptible={interruptible}, error={error}")

    # Build infrastructure
    infrastructure = Infrastructure()
    node = Cloud()
    infrastructure.add_node(node)

    # Run experiment(s)
    if error:
        adhoc_args = ((node, jobs, ci, error, seed, forecast_method, interruptible) for seed in
                      range(ERROR_REPETITIONS))
        with Pool(ERROR_REPETITIONS) as pool:  # from 0 to +-8h in 30min intervals (16 experiments)
            worker_results = pool.starmap(adhoc_worker, adhoc_args)
            timeline = np.mean(worker_results, axis=0)
    else:
        timeline = adhoc_worker(node, jobs, ci, error, None, forecast_method, interruptible)

    # _print_analysis(timeline, ci)

    # Store results
    df = pd.DataFrame({"active_jobs": timeline, "ci": ci, "emissions": ci * timeline}, index=ci.index)
    i = "_i" if interruptible else ""
    e = f"_{error}" if error else ""
    result_dir = f"results/"
    makedirs(result_dir, exist_ok=True)
    with open(f"results/{experiment_name}_{forecast_method}{e}{i}_{country}.csv", "w") as csvfile:
        df.to_csv(csvfile)


def generate_batch_jobs(n: int,
                        min_duration: int,
                        max_duration: int,
                        workday_start: int,
                        workday_end: int,
                        workdays_only: bool) -> List[Job]:
    rng = np.random.default_rng(0)
    steps = pd.date_range("2020-01-01", "2021-01-01", freq="30min")

    possible_durations = range(min_duration, max_duration, MEASUREMENT_INTERVAL)
    possible_start_times = range(workday_start, workday_end, MEASUREMENT_INTERVAL)

    days = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    if workdays_only:
        usable_days = [day for day in days if 0 <= day.weekday() <= 4]
    else:
        usable_days = days
    jobs_per_day = rng.multinomial(n, [1 / len(usable_days)] * len(usable_days))
    s = pd.Series(jobs_per_day, index=usable_days).reindex(days, fill_value=0)
    start_dates = []
    for index, value in s.items():
        ds = [index] * value
        start_minutes = np.sort(rng.choice(possible_start_times, size=len(ds)))
        for d, start_minute in zip(ds, start_minutes):
            dt = d + timedelta(minutes=int(start_minute))
            start_dates.append(steps.get_loc(dt) * MEASUREMENT_INTERVAL)

    durations = rng.choice(possible_durations, size=n)

    jobs = []
    for job_id, (start_date, duration) in enumerate(zip(start_dates, durations)):
        jobs.append((job_id, start_date, duration))
    return jobs


def _load_dataset(filename):
    with open(filename, "r") as csvfile:
        ci = pd.read_csv(csvfile, index_col=0, parse_dates=True)["Carbon Intensity"]
        logger.info(f"Average CO2 intensity of the year is {ci.mean():.2f} gCO2e/kWh.")
    return ci["2020-01-01 00:00:00":]


def _apply_error(ci, error, seed):
    if error is None:
        return ci
    rng = np.random.default_rng(seed)
    return ci + rng.normal(0, error * ci.mean(), size=len(ci))


if __name__ == '__main__':
    # Run Ad-hoc experiment
    logger.info("Starting ML Experiment using Ad-hoc strategy...")
    ml_experiment()
