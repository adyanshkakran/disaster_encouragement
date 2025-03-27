import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SEI3RD_Animation:
    def __init__(self, num_people, steps, results):
        if isinstance(num_people, np.ndarray):
            self.num_people = int(num_people[0])  # Convert array to scalar
        else:
            self.num_people = int(num_people)
        self.results = results
        self.steps = steps
        self.positions = np.random.rand(self.num_people, 2)
        self.states = np.zeros(self.num_people)  # 0: S, 1: E, 2a: I3a, 2s: I3s, 2v: I3v, 3: R, 4: D
        self.colors = {0: 'blue', 1: 'orange', 2: 'red', 3: 'darkred', 4: 'purple', 5: 'green', 6: 'black'}
        x_start = self.results[0][1]
        # Initialize states from x_start
        E = int(x_start[1])
        I3a = int(x_start[2])
        I3s = int(x_start[3])
        I3v = int(x_start[4])
        R = int(x_start[5])
        D = int(x_start[6])
        self.states[:E] = 1
        self.states[E:E+I3a] = 2
        self.states[E+I3a:E+I3a+I3s] = 3
        self.states[E+I3a+I3s:E+I3a+I3s+I3v] = 4
        self.states[E+I3a+I3s+I3v:E+I3a+I3s+I3v+R] = 5
        self.states[E+I3a+I3s+I3v+R:E+I3a+I3s+I3v+R+D] = 6
        # # Solve ODE to get f_vec for each step
        # self.simulation_results = frames
        self.fig, self.ax = plt.subplots()
        self.sc = self.ax.scatter(self.positions[:, 0], self.positions[:, 1], c=[self.colors[int(s)] for s in self.states])

        # Add Step Counter
        self.step_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=12, color='black',
                                      bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        self.legend_text = self.ax.text(1.01, 1, '', transform=self.ax.transAxes, fontsize=10, verticalalignment='center')



    def closest_individuals(self, target_state, reference_state, num):
        """Find num closest individuals with target_state to any reference_state individual."""
        reference_positions = self.positions[np.isin(self.states, reference_state)]
        target_positions = np.where(self.states == target_state)[0]

        if len(reference_positions) == 0 or len(target_positions) == 0:
            return []

        distances = []
        for i in target_positions:
            min_dist = np.min(np.linalg.norm(reference_positions - self.positions[i], axis=1))
            distances.append((i, min_dist))
        
        distances.sort(key=lambda x: x[1])
        if num > 0:
            return [i for i, _ in distances[:num]]
        else:
            return []

    def update(self, frame):

        self.positions += (np.random.rand(self.num_people, 2) - 0.5) * 0.01
        # Ensure positions remain within [0, 1]
        self.positions = np.clip(self.positions, 0, 1)

        x = self.results[frame*100][1]
        # Convert x to int
        x = x.astype(int)
        
        if frame == 0:
            f_vec = x - x
        else:
            f_vec = x - self.results[(frame - 1)*100][1].astype(int)
        # Apply transitions using f_vec
        ΔS, ΔE, ΔI3a, ΔI3s, ΔI3v, ΔR, ΔD = map(int, f_vec)
        # print(f"ΔS: {ΔS}, ΔE: {ΔE}, ΔI3a: {ΔI3a}, ΔI3s: {ΔI3s}, ΔI3v: {ΔI3v}, ΔR: {ΔR}, ΔD: {ΔD}")

        # I3 → D
        infected_indices = np.where(self.states == 4)[0]
        to_die = np.random.choice(infected_indices, size=min(ΔD, len(infected_indices)), replace=False)
        self.states[to_die] = 6  # Dead
        ΔI3v += len(to_die)

        # I3 → R
        infected_indices = np.where(np.isin(self.states, [2, 3, 4]))[0]
        to_recover = np.random.choice(infected_indices, size=min(ΔR, len(infected_indices)), replace=False)
        self.states[to_recover] = 5  # Recovered
        t1 = len(to_recover) // 3
        t2 = len(to_recover) // 4
        ΔI3s += t1
        ΔI3v += t2
        ΔI3a += len(to_recover) - t1 - t2


        # E → I3a, I3s, I3v
        exposed_indices = np.where(self.states == 1)[0]
        infected_add = abs(ΔS) - ΔE
        if len(exposed_indices) >= (infected_add):
            self.states[exposed_indices[:ΔI3a]] = 2  # Asymptomatic
            self.states[exposed_indices[ΔI3a:ΔI3a + ΔI3s]] = 3  # Symptomatic
            self.states[exposed_indices[ΔI3a + ΔI3s:ΔI3a + ΔI3s + ΔI3v]] = 4  # Severe
        ΔE += infected_add

        # S → E
        to_expose = self.closest_individuals(0, [2, 3, 4], abs(ΔS))
        for i in to_expose:
            self.states[i] = 1

        # Update scatter plot
        self.sc.set_offsets(self.positions)
        self.sc.set_color([self.colors[int(s)] for s in self.states])
        self.step_text.set_text(f"Day: {frame + 1}/{self.steps}")

        # Update legend with state counts
        state_labels = {0: "S", 1: "E", 2: "I3a", 3: "I3s", 4: "I3v", 5: "R", 6: "D"}
        state_counts = {state_labels[k]: np.sum(self.states == k) for k in range(7)}
        # legend_text = "\n".join(
        #     [f"{state_labels[k]}: {state_counts[state_labels[k]]} ({self.colors[k]})" for k in range(7)]
        # )
        legend_text = "\n".join([f"{k}: {v}" for k, v in state_counts.items()])
        self.legend_text.set_text(legend_text)
        return self.sc, self.step_text, self.legend_text

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=self.steps, interval=100, repeat=False)
        plt.show()
        
    def save_gif(self, filename='animation.gif'):
        ani = animation.FuncAnimation(self.fig, self.update, frames=self.steps, interval=100, repeat=False)
        ani.save(filename, writer='imagemagick', fps=5)
        plt.close()