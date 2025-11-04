import numpy as np
import matplotlib.pyplot as plt
import time

class JobScheduling:
    
    def __init__(self, jobs, deadlines, profits):
        self.jobs = jobs
        self.deadlines = deadlines
        self.profits = profits
        self.n = len(jobs)
        
    def solve_greedy(self):
        start_time = time.time()
        
        job_data = list(zip(self.jobs, self.deadlines, self.profits))
        job_data.sort(key=lambda x: x[2], reverse=True)
        
        max_deadline = max(self.deadlines)
        slots = [-1] * max_deadline
        scheduled_jobs = []
        total_profit = 0
        
        for job, deadline, profit in job_data:
            for slot in range(min(deadline, max_deadline) - 1, -1, -1):
                if slots[slot] == -1:
                    slots[slot] = job
                    scheduled_jobs.append((job, slot + 1, profit))
                    total_profit += profit
                    break
        
        end_time = time.time()
        
        return scheduled_jobs, total_profit, slots, end_time - start_time
    
    def print_solution(self):
        scheduled, profit, slots, exec_time = self.solve_greedy()
        
        print(f"Job Scheduling Problem - Greedy Algorithm")
        print(f"Total Jobs: {self.n}")
        print(f"Maximum Profit: {profit}")
        print(f"Jobs Scheduled: {len(scheduled)}")
        print(f"Execution Time: {exec_time:.6f} seconds\n")
        
        print("Scheduled Jobs:")
        print("-" * 50)
        for job, slot, profit in scheduled:
            print(f"Job {job}: Time Slot {slot}, Profit = {profit}")
        
        print("\nTime Slot Allocation:")
        print("-" * 50)
        for i, job in enumerate(slots):
            if job != -1:
                print(f"Slot {i+1}: Job {job}")
            else:
                print(f"Slot {i+1}: Empty")
        
        return scheduled, profit, slots
    
    def visualize_schedule(self):
        scheduled, profit, slots, exec_time = self.solve_greedy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        job_names = [s[0] for s in scheduled]
        job_profits = [s[2] for s in scheduled]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(job_names)))
        ax1.bar(job_names, job_profits, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Jobs', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Profit', fontsize=12, fontweight='bold')
        ax1.set_title(f'Profit Distribution (Total: {profit})', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        time_slots = list(range(1, len(slots) + 1))
        slot_jobs = []
        slot_colors = []
        
        for i, job in enumerate(slots):
            if job != -1:
                slot_jobs.append(job)
                job_index = self.jobs.index(job)
                slot_colors.append(colors[scheduled.index((job, i+1, self.profits[job_index]))])
            else:
                slot_jobs.append('')
                slot_colors.append('lightgray')
        
        bars = ax2.barh(time_slots, [1]*len(time_slots), color=slot_colors, 
                        edgecolor='black', linewidth=1.5)
        
        for i, (slot, job) in enumerate(zip(time_slots, slot_jobs)):
            if job != '':
                ax2.text(0.5, slot, f'Job {job}', ha='center', va='center', 
                        fontweight='bold', fontsize=10)
        
        ax2.set_yticks(time_slots)
        ax2.set_yticklabels([f'Slot {i}' for i in time_slots])
        ax2.set_xlabel('Time Unit', fontsize=12, fontweight='bold')
        ax2.set_title('Job Schedule Timeline', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class ActivitySelection:
    
    def __init__(self, activities, start_times, finish_times):
        self.activities = activities
        self.start_times = start_times
        self.finish_times = finish_times
        self.n = len(activities)
    
    def solve_greedy(self):
        start_time = time.time()
        
        activity_data = list(zip(self.activities, self.start_times, self.finish_times))
        activity_data.sort(key=lambda x: x[2])
        
        selected = [activity_data[0]]
        last_finish = activity_data[0][2]
        
        for i in range(1, len(activity_data)):
            if activity_data[i][1] >= last_finish:
                selected.append(activity_data[i])
                last_finish = activity_data[i][2]
        
        end_time = time.time()
        
        return selected, end_time - start_time
    
    def print_solution(self):
        selected, exec_time = self.solve_greedy()
        
        print(f"Activity Selection Problem - Greedy Algorithm")
        print(f"Total Activities: {self.n}")
        print(f"Maximum Activities Selected: {len(selected)}")
        print(f"Execution Time: {exec_time:.6f} seconds\n")
        
        print("Selected Activities:")
        print("-" * 60)
        print(f"{'Activity':<12} {'Start Time':<15} {'Finish Time':<15}")
        print("-" * 60)
        for activity, start, finish in selected:
            print(f"{activity:<12} {start:<15} {finish:<15}")
        
        return selected
    
    def visualize_selection(self):
        selected, exec_time = self.solve_greedy()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        all_activities = list(zip(self.activities, self.start_times, self.finish_times))
        
        for i, (activity, start, finish) in enumerate(all_activities):
            if (activity, start, finish) in selected:
                color = 'green'
                alpha = 0.8
                label = 'Selected' if i == 0 else ''
            else:
                color = 'red'
                alpha = 0.3
                label = 'Not Selected' if i == len(all_activities) - 1 else ''
            
            ax.barh(activity, finish - start, left=start, height=0.5, 
                   color=color, alpha=alpha, edgecolor='black', linewidth=1.5)
            ax.text((start + finish) / 2, activity, f'{activity}', 
                   ha='center', va='center', fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Activities', fontsize=12, fontweight='bold')
        ax.set_title(f'Activity Selection ({len(selected)}/{self.n} activities)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        green_patch = plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.8, edgecolor='black')
        red_patch = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.3, edgecolor='black')
        ax.legend([green_patch, red_patch], ['Selected', 'Not Selected'], loc='upper right')
        
        plt.tight_layout()
        plt.show()


print("="*70)
print("GREEDY ALGORITHM MINI PROJECT")
print("="*70)

print("\n### Example 1: Job Scheduling with Deadlines ###\n")
jobs = ['A', 'B', 'C', 'D', 'E']
deadlines = [2, 1, 2, 1, 3]
profits = [100, 19, 27, 25, 15]

scheduler = JobScheduling(jobs, deadlines, profits)
scheduler.print_solution()
print()
scheduler.visualize_schedule()

print("\n" + "="*70)
print("\n### Example 2: Activity Selection Problem ###\n")
activities = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
start_times = [1, 3, 0, 5, 8, 5]
finish_times = [2, 4, 6, 7, 9, 9]

activity_selector = ActivitySelection(activities, start_times, finish_times)
activity_selector.print_solution()
print()
activity_selector.visualize_selection()

print("\n" + "="*70)
print("\n### Example 3: Larger Job Scheduling Problem ###\n")
jobs_large = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8']
deadlines_large = [4, 1, 1, 1, 3, 2, 1, 3]
profits_large = [70, 60, 50, 40, 30, 20, 10, 80]

scheduler_large = JobScheduling(jobs_large, deadlines_large, profits_large)
scheduler_large.print_solution()
print()
scheduler_large.visualize_schedule()

print("\n" + "="*70)
print("\n### Example 4: Performance Comparison ###\n")
np.random.seed(42)
sizes = [10, 50, 100, 200, 500]

print(f"{'Problem Size':<15} {'Execution Time (ms)':<25} {'Max Profit':<15}")
print("-" * 55)

for size in sizes:
    jobs_perf = [f'J{i+1}' for i in range(size)]
    deadlines_perf = np.random.randint(1, size//2 + 1, size).tolist()
    profits_perf = np.random.randint(10, 200, size).tolist()
    
    scheduler_perf = JobScheduling(jobs_perf, deadlines_perf, profits_perf)
    _, profit, _, exec_time = scheduler_perf.solve_greedy()
    
    print(f"{size:<15} {exec_time*1000:<25.4f} {profit:<15}")

*** Job Scheduling with Deadlines
Total Jobs: 5

Maximum Profit: 142

Jobs Scheduled: 3

Execution Time: 0.000028 seconds

Scheduled Jobs:

Job A: Time Slot 2, Profit = 100

Job C: Time Slot 1, Profit = 27

Job E: Time Slot 3, Profit = 15

Time Slot Allocation:

Slot 1: Job C

Slot 2: Job A

Slot 3: Job E

Example 2: Activity Selection Problem
Total Activities: 6

Maximum Activities Selected: 4

Execution Time: 0.000014 seconds

Selected Activities:

text
Activity     Start Time      Finish Time    
------------------------------------------------------------
A1           1               2              
A2           3               4              
A4           5               7              
A5           8               9
***
