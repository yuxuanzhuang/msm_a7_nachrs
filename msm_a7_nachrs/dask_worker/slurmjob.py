import socket
import subprocess
from datetime import datetime


class SLURMJob(object):
    def __init__(self, n_nodes, ip_address, port):
        self.n_nodes = n_nodes
        self.ip_address = ip_address
        self.port = str(port)

        self._generate_job()

    def _generate_job(self):
        # specfic slurm script in different cluster
        raise NotImplementedError('Only for inheritence')

    def submit_job(self):
        # specfic slurm script in different cluster
        raise NotImplementedError('Only for inheritence')

    def check_status(self):
        proc = subprocess.Popen(['scontrol', 'show', 'jobid', '-dd', self.jobid],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        out, err = out.decode(), err.decode()
        if proc.returncode != 0:
            raise RuntimeError(
                "Command exited with non-zero exit code.\n"
                "Exit code: {}\n"
                "stdout:\n{}\n"
                "stderr:\n{}\n".format(proc.returncode, out, err)
            )
        self.out = out
        self.err = err
        self.runtime = out[out.find('RunTime') + 8: out.find('RunTime') + 16]
        self.runtime = datetime.strptime(
            self.runtime, '%H:%M:%S') - datetime(1900, 1, 1)
        print(self.runtime)

    def cancel_worker(self):
        proc = subprocess.Popen(
            ['scancel', self.jobid], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        out, err = out.decode(), err.decode()

        if proc.returncode != 0:
            raise RuntimeError(
                "Command exited with non-zero exit code.\n"
                "Exit code: {}\n"
                "stdout:\n{}\n"
                "stderr:\n{}\n".format(proc.returncode, out, err)
            )

        print('job is canceled')

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        self.cancel_worker()
        return False

def add_workers(n_nodes, slurmjob, ip_address, port):
    """
    Add dask workers in Lindahl TCBLAB CLuster
    Parameters
    ----------
    n_nodes: int
        number of nodes
    slurmjob: SLURMJob
        SLURMJob class
    ip_address: str
        ip address of the dask scheduler
    port: int
        port of the dask scheduler
    """
    slurm_job = slurmjob(n_nodes=n_nodes,
                        ip_address=ip_address,
                        port=port)
    slurm_job.submit_job()
    return slurm_job