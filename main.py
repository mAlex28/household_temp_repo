from models.agent import Household
from models.model import HouseholdEnergyModel
from q_learning_agent import QLearningAgent
from energy_environment import EnergyEnvironment
import numpy as np
from graph_generator import generate_comparison_graphs

class EnergyModel:
    def __init__(self, num_households, num_rooms, season):
        self.num_households = num_households
        self.num_rooms = num_rooms
        self.season = season
        self.household_model = HouseholdEnergyModel(num_households, season)
        
        # Updated to include 5 actions (light, washing_machine, fridge, gas_heating, gas_cooking)
        self.q_learning_agent = QLearningAgent(state_size=[2, 2, 2, 2, 2], action_size=[2, 2, 2, 2, 2])

    def train_agent(self, episodes=2000):
        env = EnergyEnvironment(num_rooms=self.num_rooms, season=self.season)

        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.q_learning_agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.q_learning_agent.learn(state, action, reward, next_state)
                state = next_state

    def test_agent_exploitation(self):
        env = EnergyEnvironment(num_rooms=self.num_rooms, season=self.season)
        state = env.reset()
        total_energy_used = 0
        total_gas_used = 0
        total_cost = 0

        for _ in range(90):  # Simulate for 90 days
            for hour in range(24):
                action = self.q_learning_agent.choose_action(state)  # Exploit best known action
                state, reward, done, _ = env.step(action)
                energy_used = -reward / (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                cost = energy_used * (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                
                # Simulate gas usage if heating or cooking is active
                gas_used = sum(action[-2:]) * env.appliance_usage.get('gas_heating', 0) + sum(action[-1:]) * env.appliance_usage.get('gas_cooking', 0)
                total_energy_used += energy_used
                total_gas_used += gas_used
                total_cost += cost + (gas_used * 0.04)  # Assuming 0.04 £ per unit of gas

        return total_energy_used, total_gas_used, total_cost

    def run(self):
        print(f"\nTraining agent for {self.season} season...")
        self.train_agent()
        print(f"Testing agent for {self.season} season...")

        # Test agent performance
        total_electricity_usage_trained, total_gas_usage_trained, total_cost_trained = self.test_agent_exploitation()

        # Test random policy performance
        total_electricity_usage_random, total_gas_usage_random, total_cost_random = self.test_random_policy()

        # Print results
        print("\nResults with trained agent:")
        print(f"Electricity and Gas Usage and Cost for {self.season.capitalize()}")
        print(f"Total Electricity Usage: {total_electricity_usage_trained} kWh\n"
              f"Total Gas Usage: {total_gas_usage_trained} units\n"
              f"Total Cost: £{total_cost_trained:.2f}\n")

        print("\nResults with random policy:")
        print(f"{self.season.capitalize()} - Total Electricity Usage: {total_electricity_usage_random} kWh\n"
              f"Total Gas Usage: {total_gas_usage_random} units\n"
              f"Total Cost: £{total_cost_random:.2f}\n")

        print("\nPerformance improvement with trained agent:")
        print(f"{self.season.capitalize()} - Total Electricity Usage Reduction (vs Random): {total_electricity_usage_random - total_electricity_usage_trained} kWh\n"
              f"Total Gas Usage Reduction (vs Random): {total_gas_usage_random - total_gas_usage_trained} units\n"
              f"Total Cost Reduction (vs Random): £{total_cost_random - total_cost_trained:.2f}\n")
        
        # Generate and save comparison graphs
        generate_comparison_graphs(self.season, total_electricity_usage_trained, total_gas_usage_trained, total_cost_trained,
                                    total_electricity_usage_random, total_gas_usage_random, total_cost_random)

    def test_random_policy(self):
        env = EnergyEnvironment(num_rooms=self.num_rooms, season=self.season)
        state = env.reset()
        total_energy_used = 0
        total_gas_used = 0
        total_cost = 0

        for _ in range(90):  # Simulate for 90 days
            for hour in range(24):
                action = [np.random.choice([0, 1]) for _ in range(len(state))]
                state, reward, done, _ = env.step(action)
                energy_used = -reward / (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                cost = energy_used * (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                
                # Simulate gas usage
                gas_used = sum(action[-2:]) * env.appliance_usage.get('gas_heating', 0) + sum(action[-1:]) * env.appliance_usage.get('gas_cooking', 0)
                total_energy_used += energy_used
                total_gas_used += gas_used
                total_cost += cost + (gas_used * 0.04)  # Assuming 0.04 £ per unit of gas

        return total_energy_used, total_gas_used, total_cost


if __name__ == "__main__":
    num_households = 100
    num_rooms = 3
    seasons = ['winter', 'summer']

    for season in seasons:
        print(f"Running model for {season}...")
        model = EnergyModel(num_households, num_rooms, season=season)
        model.run()
