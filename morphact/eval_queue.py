"""Claim episodes across simulator processes without changing case identities."""

import fcntl
import itertools


def task_episodes(task_ids, trials, worker, workers, queue_path=None):
    jobs = [(task, episode) for task in task_ids for episode in range(trials)]

    def claim_jobs():
        if queue_path is None:
            yield from jobs[worker::workers]
            return
        with open(queue_path, "a+") as cursor:
            while True:
                fcntl.flock(cursor, fcntl.LOCK_EX)
                try:
                    cursor.seek(0)
                    index = int(cursor.read() or "0")
                    if index >= len(jobs):
                        return
                    cursor.seek(0)
                    cursor.truncate()
                    cursor.write(str(index + 1))
                    cursor.flush()
                finally:
                    fcntl.flock(cursor, fcntl.LOCK_UN)
                yield jobs[index]

    for task, group in itertools.groupby(claim_jobs(), key=lambda job: job[0]):
        yield task, (episode for _, episode in group)
