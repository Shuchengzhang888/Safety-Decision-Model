import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray

class HighwayEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsTraj"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "simulation_frequency": 10,
            'policy_frequency': 2,
            "lanes_count": 4,
            "vehicles_count": 30,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 120,  # [s]
            "ego_spacing": 1,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "high_speed_reward": 0.6,  # The reward received when driving at high speed
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [25, 40],
            "offroad_terminal": False,
            "show_trajectories": True,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        #Create a road composed of straight adjacent lanes.
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30, length=1300.0),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        #Create some new random vehicles of a given type, and add them on the road.
        ######################
        # Use reference model
        reference = False
        #######################

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            if not reference:
                vehicle =Vehicle.create_random(
                    self.road,
                    speed=45,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"]
                )
                vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            else:
                vehicle =other_vehicles_type.create_random(
                self.road,
                speed=45,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.MAX_SPEED -= 25
                vehicle.MIN_SPEED += 25
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        #The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        #The episode is over if the ego vehicle crashed or the time is out or length is out.
        ######################
        test_length = 1200
        ######################
        if self.vehicle.position[0] >= test_length:
            return True
        return self.vehicle.crashed or \
            self.time >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        #The cost signal is the occurrence of collision.

        return float(self.vehicle.crashed)

register(
    id='my-highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)
