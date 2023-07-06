import numpy as np
import os
from chencrafts.toolbox.optimize import Optimization, MultiOpt, OptTraj, MultiTraj

def target_func(coord, offset=[0, 0, 0]):
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
    def create_opt(self):
        opt = Optimization(
            fixed_variables=fixed_params,
            free_variable_ranges=bounds,
            target_func=target_func,
            target_kwargs={"offset": offset},
        )
        return opt

    def test_opt(self):
        opt = self.create_opt()
        traj = opt.run(
            file_name = f"{data_path}/traj.csv", 
            fixed_para_file_name = f"{data_path}/traj_fixed.csv"
        )

        assert np.abs(traj.final_full_para["x1"] + offset[0]) < 1e-5
        assert np.abs(traj.final_full_para["x2"] + offset[1]) < 1e-5

        loaded_traj = OptTraj.from_file(f"{data_path}/traj.csv", f"{data_path}/traj_fixed.csv")

        assert loaded_traj.final_full_para["x1"] == traj.final_full_para["x1"]
        assert loaded_traj.final_full_para["x2"] == traj.final_full_para["x2"]
        assert loaded_traj.final_full_para["x3"] == traj.final_full_para["x3"]

    def test_save_traj(self):
        """
        Must be run after test_opt
        """
        opt = self.create_opt()
        traj = opt.run()

        traj.save(f"{current_path}/data/traj.csv", f"{data_path}/traj_fixed.csv")

        loaded_traj = OptTraj.from_file(f"{data_path}/traj.csv", f"{data_path}/traj_fixed.csv")
        assert loaded_traj.final_full_para["x1"] == traj.final_full_para["x1"]
        assert loaded_traj.final_full_para["x2"] == traj.final_full_para["x2"]
        assert loaded_traj.final_full_para["x3"] == traj.final_full_para["x3"]

    # ##############################################################################
    def create_multi_opt(self):
        multi_opt = MultiOpt(
            self.create_opt(),
        )
        return multi_opt
    
    def test_multi_opt(self):
        multi_path = f"{data_path}/multi_traj/"
        if not os.path.exists(multi_path):
            os.mkdir(multi_path)

        multi_opt = self.create_multi_opt()
        multi_traj = multi_opt.run(
            3,
            save_path=multi_path,
        )

        para = multi_traj.best_traj().final_full_para
        assert np.abs(para["x1"] + offset[0]) < 1e-5
        assert np.abs(para["x2"] + offset[1]) < 1e-5

        loaded_multi_traj = MultiTraj.from_folder(multi_path)
        loaded_para = loaded_multi_traj.best_traj().final_full_para

        assert loaded_para["x1"] == para["x1"]
        assert loaded_para["x2"] == para["x2"]
        assert loaded_para["x3"] == para["x3"]