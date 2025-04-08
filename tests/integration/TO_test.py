import torch
from pyfreefem import FreeFemRunner
import yaml
import numpy as np
import time
import matplotlib.pyplot as plt

class MLaSO_d:
    def __init__(self):
        with open("./config/MLaSO-d_param.yaml", "r") as yml:
            config = yaml.safe_load(yml)

        self.beta_min = config["MLaSO_d"]["beta_min"]
        self.k = config["MLaSO_d"]["k"]
        self.ks = config["MLaSO_d"]["ks"]
        self.kr = config["MLaSO_d"]["kr"]
        self.kp = config["MLaSO_d"]["kp"]
        self.max_loop = config["MLaSO_d"]["max_loop"]
        self.Ne = config["MLaSO_d"]["Ne"]
        self.r_min = config["MLaSO_d"]["r_min"]

        self.max_epoch = config["on_the_job_training"]["max_epoch"]
        self.gamma = config["on_the_job_training"]["gamma_0"]
        self.delta_gamma = config["on_the_job_training"]["delta_gamma"]
        self.R_min = config["on_the_job_training"]["R_min"]
        self.E_min = config["on_the_job_training"]["E_min"]

        self.rho = torch.ones([self.Ne], dtype=torch.float32)
        self.d = torch.zeros([self.Ne], dtype=torch.float32)
        
        self.lamG = 0.1

        self.comp = []

    def on_the_job_training(self) -> None:
        run_fem = FreeFemRunner("./src/freefem/2Delastic-FEA.edp")
        rho_np = self.rho.numpy()
        print('input rho : ',rho_np,' rho_min : ',np.min(rho_np),' rho_max : ',np.max(rho_np),' rho_ave : ',np.mean(rho_np))
        run_fem.import_variables(
            E=210e9,
            nu=0.31,
            Volcoef=0.5,
            lamG=self.lamG,
            SigG=0.03,
            rhomin=0.001,
            rhomax=1.0,
            regR=0.04,
            kr=float(self.kr),
            rho=rho_np,
        )  # 1
        result = run_fem.execute()

        self.d = torch.tensor(result["d"],dtype=torch.float32)  # 2
        # print("freefem sensitivity : ",self.d)
        self.comp.append(result["comp"])
        self.lamG = result["lamG"]
        # MLaSO-dの実行

    def implementation(self):

        dt = 0.1

        for self.k in range(self.max_loop):
            print('ITER : ',self.k)
            print("previous rho : ",self.rho,' rho_min : ',self.rho.min(),' rho_max : ',self.rho.max(),' rho_ave : ',torch.mean(self.rho))
            
            self.on_the_job_training()
            
            self.rho = torch.clamp(self.rho + dt * self.d, min=0.001, max=1.0)
            
            print("updated rho : ",self.rho,' rho_min : ',self.rho.min(),' rho_max : ',self.rho.max(),' rho_ave : ',torch.mean(self.rho))
            print("sensitivity : ",self.d,' sensitivity_min : ',self.d.min(),' sensitivity_max : ',self.d.max())
            
            print('comp',self.comp[-1])
            
            if self.k%10==0:
                save_fem = FreeFemRunner(f'./src/freefem/save_vtk.edp')
                save_fem.import_variables(rho=self.rho.numpy(),k=float(self.k))
                save_fem.execute()
        
        x=np.arange(0,self.max_loop)
        plt.plot(x,self.comp)
        plt.show()
        


if __name__ == "__main__":
    start=time.time()
    mlaso = MLaSO_d()
    mlaso.implementation()
    end=time.time()
    print(f'duration:{end-start}') 