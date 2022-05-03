#  add dask workers in Lindahl TCBLAB CLuster
import netifaces as ni
import subprocess
from .slurmjob import SLURMJob

import dask

dask_worker_dir = '/'.join(dask.__file__.split('/')[:-5]) + '/bin/dask-worker '
#  For tcblab cluster
try:
    ni.ifaddresses('enp24s0')
    ip_address = ni.ifaddresses('enp24s0')[ni.AF_INET][0]['addr']
except:
    pass
try:
    ni.ifaddresses('enp5s0')
    ip_address = ni.ifaddresses('enp5s0')[ni.AF_INET][0]['addr']
except:
    pass
try:
    ni.ifaddresses('enp4s0')
    ip_address = ni.ifaddresses('enp4s0')[ni.AF_INET][0]['addr']
except:
    pass
try:
    ni.ifaddresses('enp3s0')
    ip_address = ni.ifaddresses('enp3s0')[ni.AF_INET][0]['addr']
except:
    pass
try:
    ni.ifaddresses('enp2s0f0')
    ip_address = ni.ifaddresses('enp2s0f0')[ni.AF_INET][0]['addr']
except:
    pass
try:
    ni.ifaddresses('ens6')
    ip_address = ni.ifaddresses('ens6')[ni.AF_INET][0]['addr']
except:
    pass

port = 8786

class SLURMJob(SLURMJob):
    def _generate_job(self):
        self.jobscript = []
        self.jobscript.append("#!/usr/bin/env bash \n")
        self.jobscript.append("#SBATCH -J dask-worker\n")
#        self.jobscript.append("#SBATCH -p lindahl\n")
        self.jobscript.append("#SBATCH -p tcb\n")

        self.jobscript.append("#SBATCH --nodes=" + str(self.n_nodes) + '\n')
        self.jobscript.append("#SBATCH --cpus-per-task=1\n")
        self.jobscript.append("#SBATCH --ntasks-per-core=1\n")
        self.jobscript.append("#SBATCH --ntasks-per-node=8\n")
        self.jobscript.append("#SBATCH -t 24:00:00\n")
        self.jobscript.append("#SBATCH -C cpu\n")
        self.jobscript.append("\n")
        self.jobscript.append("srun " + dask_worker_dir + self.ip_address + ":" + self.port + " --memory-limit=auto --nthreads 2 --nprocs 1 --death-timeout 60\n")
        
    def submit_job(self):
        with open('/nethome/yzhuang/dask-worker-space/submit.sh', "w") as f:
            f.writelines(self.jobscript)    
        proc = subprocess.Popen(
            ['sbatch', 'submit.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    cwd='/nethome/yzhuang/dask-worker-space'
        )

        out, err = proc.communicate()
        out, err = out.decode(), err.decode()

        if proc.returncode != 0:
            raise RuntimeError(
                "Command exited with non-zero exit code.\n"
                "Exit code: {}\n"
                "stdout:\n{}\n"
                "stderr:\n{}\n".format(proc.returncode, out, err)
            )

        self.jobid = out.split(' ')[-1][:-1]
    

def add_workers(n_nodes, ip_address=ip_address, port=port):
    slurm_job = SLURMJob(n_nodes=n_nodes,
                         ip_address=ip_address,
                         port=port)
    slurm_job.submit_job()
    return slurm_job
