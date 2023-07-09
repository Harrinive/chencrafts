import numpy as np
import os
from chencrafts.toolbox.optimize import Optimization, MultiOpt, OptTraj, MultiTraj
import time

def target_func(coord, offset=[0, 0, 0], error_prob=0.0):
    if np.random.rand() < error_prob:
        raise ValueError("Random error occurs")
    return (
        (coord["x1"] + offset[0])**2 
        + (coord["x2"] + offset[1])**2 
        + (coord["x3"] + offset[2])**2
    )

offset = [-1, 2, 1]
fixed_params = {"x3": -1}
bounds = {"x1": (-10, 10), "x2": (-10, 10)}

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = f"{current_path}/data"
# if /data not exist, create it
if not os.path.exists(data_path):
    os.mkdir(data_path)

class TestOpt():
    # ##############################################################################
    def create_opt(self, error_prob=0.0):
        opt = Optimization(
            fixed_variables=fixed_params,
            free_variable_ranges=bounds,
            target_func=target_func,
            target_kwargs={"offset": offset, "error_prob": error_prob},
        )
        return opt

    def test_opt(self):
        opt = self.create_opt(error_prob=0.0)
        traj = opt.run(
            file_name = f"{data_path}/traj.csv", 
            fixed_para_file_name = f"{data_path}/traj_fixed.csv"
        )

        assert np.abs(traj.final_full_para["x1"] + offset[0]) < 1e-5
        assert np.abs(traj.final_full_para["x2"] + offset[1]) < 1e-5

        loaded_traj = OptTraj.from_file(f"{data_path}/traj.csv", f"{data_path}/traj_fixed.csv")
        assert np.abs(loaded_traj.final_full_para["x1"] - traj.final_full_para["x1"]) < 1e-15
        assert np.abs(loaded_traj.final_full_para["x2"] - traj.final_full_para["x2"]) < 1e-15
        assert np.abs(loaded_traj.final_full_para["x3"] - traj.final_full_para["x3"]) < 1e-15

        traj.save(f"{current_path}/data/traj.csv", f"{data_path}/traj_fixed.csv")
        loaded_traj2 = OptTraj.from_file(f"{data_path}/traj.csv", f"{data_path}/traj_fixed.csv")
        assert np.abs(loaded_traj2.final_full_para["x1"] - traj.final_full_para["x1"]) < 1e-15
        assert np.abs(loaded_traj2.final_full_para["x2"] - traj.final_full_para["x2"]) < 1e-15
        assert np.abs(loaded_traj2.final_full_para["x3"] - traj.final_full_para["x3"]) < 1e-15

    # ##############################################################################
    def create_multi_opt(self):
        multi_opt = MultiOpt(
            self.create_opt(error_prob=0.03),
        )
        return multi_opt
    
    def create_and_clear_folder(self, multi_path):
        # create a folder to store the multi_traj
        if not os.path.exists(multi_path):
            os.mkdir(multi_path)
        # clear the folder
        for file_name in os.listdir(multi_path):
            os.remove(f"{multi_path}/{file_name}")
    
    def run_multi_opt(self):
        multi_path = f"{data_path}/multi_traj/"
        self.create_and_clear_folder(multi_path)

        multi_opt = self.create_multi_opt()
        multi_traj = multi_opt.run(
            20,
            save_path=multi_path,
            cpu_num=4
        )
        print(multi_traj.length)

        para = multi_traj.best_traj().final_full_para
        assert np.abs(para["x1"] + offset[0]) < 1e-5
        assert np.abs(para["x2"] + offset[1]) < 1e-5

        loaded_multi_traj = MultiTraj.from_folder(multi_path)
        loaded_para = loaded_multi_traj.best_traj().final_full_para

        assert loaded_multi_traj.length == multi_traj.length
        assert np.abs(para["x1"] - loaded_para["x1"]) < 1e-15
        assert np.abs(para["x2"] - loaded_para["x2"]) < 1e-15
        assert np.abs(para["x3"] - loaded_para["x3"]) < 1e-15
    
    def test_multi_opt(self):
        for _ in range(3):
            self.run_multi_opt()

if __name__ == "__main__":
    test_opt = TestOpt()
    # test_opt.test_opt()
    test_opt.test_multi_opt()