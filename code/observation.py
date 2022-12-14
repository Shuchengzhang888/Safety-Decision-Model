from collections import OrderedDict
from itertools import product
from typing import List, Dict, TYPE_CHECKING, Optional, Union, Tuple
from gym import spaces
import numpy as np
import pandas as pd

from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.envs.common.graphics import EnvViewer
from highway_env.road.lane import AbstractLane
from highway_env.utils import distance_to_circle, Vector
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

class ObservationType(object):
    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        #Get the observation space.
        raise NotImplementedError()

    def observe(self):
        #Get an observation of the environment state.
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        #The vehicle observing the scene.

        #If not set, the first controlled vehicle is used by default.

        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class KinematicTrajObservation(ObservationType):

    #Observe the kinematics of nearby vehicles with predicted trajectories

    FEATURES: List[str] = ['x', 'y', 'vx', 'vy']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = False,
                 observe_intentions: bool = False,
                 **kwargs: dict) -> None:

        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        ##########################
        #Input size
        #Including 20, 28, 36, 44
        self.input_size = 20
        ############################
    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.input_size,), low=-np.inf, high=np.inf, dtype=np.float32)

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            obss = np.array([origin.position[0], origin.position[1], origin.velocity[0], origin.velocity[1]])

            for v in close_vehicles:
                if self.input_size == 28:
                    #4 + 4*6
                    obss = np.hstack((obss,[v.position[0]-origin.position[0], v.position[1]-origin.position[1], \
                                        v.velocity[0]-origin.velocity[0], v.velocity[1]-origin.velocity[1], \
                                        v.pred_position1[0]-origin.pred_position1[0], v.pred_position1[1]-origin.pred_position1[1]]))
                elif self.input_size == 20:
                    #4 + 4*4
                    obss = np.hstack((obss,[v.position[0]-origin.position[0], v.position[1]-origin.position[1], \
                                        v.velocity[0]-origin.velocity[0], v.velocity[1]-origin.velocity[1]]))
                elif self.input_size == 36:
                    #4 + 4*8
                    obss = np.hstack((obss,[v.position[0]-origin.position[0], v.position[1]-origin.position[1], \
                                    v.velocity[0]-origin.velocity[0], v.velocity[1]-origin.velocity[1], \
                                    v.pred_position1[0]-origin.pred_position1[0], v.pred_position1[1]-origin.pred_position1[1], \
                                    v.pred_position2[0]-origin.pred_position2[0], v.pred_position2[1]-origin.pred_position2[1]]))
                elif self.input_size == 44:
                    obss = np.hstack((obss,[v.position[0]-origin.position[0], v.position[1]-origin.position[1], \
                                        v.velocity[0]-origin.velocity[0], v.velocity[1]-origin.velocity[1], \
                                        v.pred_position1[0]-origin.pred_position1[0], v.pred_position1[1]-origin.pred_position1[1], \
                                        v.pred_position2[0]-origin.pred_position2[0], v.pred_position2[1]-origin.pred_position2[1], \
                                        v.pred_position3[0]-origin.pred_position3[0], v.pred_position3[1]-origin.pred_position3[1]]))
                else:
                    pass
            while obss.shape[0] < self.input_size:
                    obss = np.hstack([obss, 0])
            return obss.astype(self.space().dtype)
