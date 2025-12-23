import random
import numpy as np
from tqdm import tqdm

class Particle:
    def __init__(self, bounds):
        self.position = {}
        self.velocity = {}
        self.best_position = {}
        self.best_score = -1.0
        
        for key, (lower, upper) in bounds.items():
            self.position[key] = random.uniform(lower, upper)
            self.velocity[key] = random.uniform(-1, 1) * (upper - lower) * 0.1
            self.best_position[key] = self.position[key]

    def update_velocity(self, global_best_pos, w=0.5, c1=1.5, c2=1.5):
        for key in self.position:
            r1 = random.random()
            r2 = random.random()
            
            cognitive = c1 * r1 * (self.best_position[key] - self.position[key])
            social = c2 * r2 * (global_best_pos[key] - self.position[key])
            
            self.velocity[key] = w * self.velocity[key] + cognitive + social

    def update_position(self, bounds):
        for key in self.position:
            self.position[key] += self.velocity[key]
            
            # Clamp to bounds
            lower, upper = bounds[key]
            if self.position[key] < lower:
                self.position[key] = lower
                self.velocity[key] *= -1 # Bounce back
            elif self.position[key] > upper:
                self.position[key] = upper
                self.velocity[key] *= -1

class PSOOptimizer:
    def __init__(self, bounds, num_particles=5, iterations=3):
        self.bounds = bounds
        self.num_particles = num_particles
        self.iterations = iterations
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_score = -1.0

    def optimize(self, fitness_function):
        print(f"Starting PSO Optimization with {self.num_particles} particles and {self.iterations} iterations.")
        
        # Initialize global best
        self.global_best_position = self.particles[0].position.copy()
        
        iter_pbar = tqdm(range(self.iterations), desc="PSO Overall Progress")
        for i in iter_pbar:
            particle_pbar = tqdm(enumerate(self.particles), total=self.num_particles, desc=f"Iteration {i+1} Particles", leave=False)
            for j, particle in particle_pbar:
                particle_pbar.set_postfix({'current_best': f"{self.global_best_score:.4f}"})
                score = fitness_function(particle.position)
                
                # Update personal best
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if score > self.global_best_score:
                    improvement = score - self.global_best_score if self.global_best_score > 0 else score
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
                    tqdm.write(f"New Global Best Found: {self.global_best_score:.4f} (Improved by {improvement:.4f})")

            # Update particles
            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position(self.bounds)
                
        return self.global_best_position, self.global_best_score
