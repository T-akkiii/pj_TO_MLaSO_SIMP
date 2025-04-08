import torch
from pyfreefem import FreeFemRunner
import yaml
import numpy as np
import time
import matplotlib.pyplot as plt

import subprocess

cmd = ["FreeFem++", "./src/freefem/sensitivity-analysis.edp"]
process = subprocess.run(cmd, text=True, capture_output=True)
print("STDOUT:", process.stdout)
print("STDERR:", process.stderr)

class MLaSO_d:
    def __init__(self):
        with open("./config/MLaSO-d-lsm-param.yaml", "r") as yml:
            config = yaml.safe_load(yml)

        self.beta_min = config["MLaSO_d"]["beta_min"]
        self.k = config["MLaSO_d"]["k"]
        self.ks = config["MLaSO_d"]["ks"]
        self.kr = config["MLaSO_d"]["kr"]
        self.kp = config["MLaSO_d"]["kp"]
        self.max_loop = config["MLaSO_d"]["max_loop"]
        self.Ne = config["MLaSO_d"]["Ne"]
        self.r_min = config["MLaSO_d"]["r_min"]

        self.phi = torch.ones([self.Ne], dtype=torch.float32)
        self.ophi = torch.ones([self.Ne], dtype=torch.float32)
        self.TD= torch.zeros([self.Ne], dtype=torch.float32)
        
        self.LagGv = 1.

        self.obj = []

    def sensitivity_analysis(self) -> None:
        run_sensitivity_analysis = FreeFemRunner("./src/freefem/sens-ana.edp")
        phi_np = self.phi.numpy()
        print('input phi : ',phi_np,' phi_min : ',np.min(phi_np),' phi_max : ',np.max(phi_np),' phi_ave : ',np.mean(phi_np))
        run_sensitivity_analysis.import_variables(
            E=210e9,
            nu=0.31,
            kr=float(self.kr),
            LagGv=self.LagGv,
            phi=phi_np
        )  # 1
        result = run_sensitivity_analysis.execute()

        self.TD = torch.tensor(result["TD"],dtype=torch.float32)  # 2
        self.obj.append(result["obj"])
        self.LagGv = result["LagGv"]
        
    def update_levelset_function(self)->None:
        run_update=FreeFemRunner("./src/freefem/reaction-difusion-equation.edp")
        run_update.import_variables(
            TD=self.TD,
            LagGv=self.LagGv,
            ophi=self.phi
        )
        result=run_update.execute()
        return result["phi"]

    def implementation(self):
        for self.k in range(self.max_loop):
            print('ITER : ',self.k)
            print("previous phi : ",self.phi,' phi_min : ',self.phi.min(),' phi_max : ',self.phi.max(),' phi_ave : ',torch.mean(self.phi))
            
            self.sensitivity_analysis()
            
            self.phi = self.update_levelset_function()
            
            print("updated phi : ",self.phi,' phi_min : ',self.phi.min(),' phi_max : ',self.phi.max(),' phi_ave : ',torch.mean(self.phi))
            print("sensitivity : ",self.TD,' sensitivity_min : ',self.TD.min(),' sensitivity_max : ',self.TD.max())
            
            print('obj',self.obj[-1])
            
            # if self.k%10==0:
            save_fem = FreeFemRunner('./src/freefem/save_vtk.edp')
            save_fem.import_variables(phi=self.phi.numpy(),
                                        k=float(self.k))
            save_fem.execute()
        
        x=np.arange(0,self.max_loop)
        plt.plot(x,self.obj)
        plt.show()

if __name__ == "__main__":
    start=time.time()
    mlaso = MLaSO_d()
    mlaso.implementation()
    end=time.time()
    print(f'duration:{end-start}') 