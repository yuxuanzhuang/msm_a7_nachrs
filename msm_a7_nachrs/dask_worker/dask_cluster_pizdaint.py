#  add dask workers in piz-daint CLuster
import socket
import subprocess
from slurmjob import SLURMJob
ip_address = socket.gethostbyname(socket.gethostname())
port = 8786

class PizDaintSLURMJob(SLURMJob):
    def _generate_job(self):
        self.jobscript = []
        self.jobscript.append("#!/usr/bin/env bash \n")
        self.jobscript.append("#SBATCH -J dask-worker\n")
        self.jobscript.append("#SBATCH -p normal\n")
        self.jobscript.append("#SBATCH -A pr89\n")
        self.jobscript.append("#SBATCH --nodes=" + str(self.n_nodes) + '\n')
        self.jobscript.append("#SBATCH --cpus-per-task=1\n")
        self.jobscript.append("#SBATCH --ntasks-per-core=1\n")
        self.jobscript.append("#SBATCH --ntasks-per-node=12\n")
        self.jobscript.append("#SBATCH -t 24:00:00\n")
        self.jobscript.append("#SBATCH -C gpu\n")
        self.jobscript.append("\n")
        self.jobscript.append("module load daint-gpu\n")
        self.jobscript.append("module load cray-python\n")
        self.jobscript.append("module load jupyter-utils\n")
        self.jobscript.append("module load PyExtensions\n")
        self.jobscript.append("module load jupyterlab/2.0.2-CrayGNU-20.11-batchspawner-cuda\n")
        self.jobscript.append("\n")
        self.jobscript.append("source /users/yzhuang/myvenv/bin/activate\n")
        self.jobscript.append("srun /users/yzhuang/myvenv/bin/dask-worker " + self.ip_address + ":" + self.port + " --nthreads 1 --nprocs 2 --death-timeout 60\n")
        
    def submit_job(self):
        with open('./dask-worker-space/submit.sh', "w") as f:
            f.writelines(self.jobscript)    
        proc = subprocess.Popen(
            ['sbatch', 'submit.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    cwd='/users/yzhuang/jupyterground/dask-worker-space'
        )

        out, err = proc.communicate()
        out, err = out.decode(), err.decode()

        if proc.returncode != 0:
            raise RuntimeError(
                "Command exited with non-zero exit code.\n"
                "Exit code: {}\n"
                "stdout:\n{}\n"
                "stderr:\n{}\n".format(proc.returncod, out, err)
            )

        self.jobid = out.split(' ')[-1][:-1]
    
    
def add_workers(n_nodes):
    slurm_job = SLURMJob(n_nodes=n_nodes,
                         ip_address=ip_address,
                         port=port)
    slurm_job.submit_job()
    return slurm_job
    