import sys
import matplotlib.pyplot as plt
from lab_amoro.parallel_robot import *
from biglide_models import *
import rclpy
import numpy as np

# Function to initialize Matplotlib figures and lines
def init_plots():
    # Initialize position plots
    fig_position, axs_position = plt.subplots(2, 2, figsize=(10, 8))
    line_x, = axs_position[0, 0].plot([], [], 'm-', label='x')
    line_x_model, = axs_position[0, 0].plot([], [], 'c-', label='x computed')
    line_y, = axs_position[1, 0].plot([], [], 'm-', label='y')
    line_y_model, = axs_position[1, 0].plot([], [], 'c-', label='y computed')
    line_q12, = axs_position[0, 1].plot([], [], 'm-', label='q12')
    line_q12_model, = axs_position[0, 1].plot([], [], 'c-', label='q12 computed')
    line_q22, = axs_position[1, 1].plot([], [], 'm-', label='q22')
    line_q22_model, = axs_position[1, 1].plot([], [], 'c-', label='q22 computed')

    axs_position[0, 0].set_title('Position x [m]')
    axs_position[1, 0].set_title('Position y [m]')
    axs_position[0, 1].set_title('Position q12 [rad]')
    axs_position[1, 1].set_title('Position q22 [rad]')

    axs_position[0, 0].set_ylim([-0.5, 0.5])
    axs_position[1, 0].set_ylim([-0.6, 6])
    axs_position[0, 1].set_ylim([-np.pi, np.pi])
    axs_position[1, 1].set_ylim([-np.pi, np.pi])

    axs_position[0, 0].legend()
    axs_position[1, 0].legend()
    axs_position[0, 1].legend()
    axs_position[1, 1].legend()

    # Initialize velocity plots
    fig_velocity, axs_velocity = plt.subplots(2, 2, figsize=(10, 8))
    line_xD, = axs_velocity[0, 0].plot([], [], 'm-', label='v_x')
    line_xD_model, = axs_velocity[0, 0].plot([], [], 'c-', label='v_x computed')
    line_yD, = axs_velocity[1, 0].plot([], [], 'm-', label='v_y')
    line_yD_model, = axs_velocity[1, 0].plot([], [], 'c-', label='v_y computed')
    line_q12D, = axs_velocity[0, 1].plot([], [], 'm-', label='q12D')
    line_q12D_model, = axs_velocity[0, 1].plot([], [], 'c-', label='q12D computed')
    line_q22D, = axs_velocity[1, 1].plot([], [], 'm-', label='q22D')
    line_q22D_model, = axs_velocity[1, 1].plot([], [], 'c-', label='q22D computed')

    axs_velocity[0, 0].set_title('Velocity x [m/s]')
    axs_velocity[1, 0].set_title('Velocity y [m/s]')
    axs_velocity[0, 1].set_title('Velocity q12 [rad/s]')
    axs_velocity[1, 1].set_title('Velocity q22 [rad/s]')

    axs_velocity[0, 0].set_ylim([-0.02, 0.02])
    axs_velocity[1, 0].set_ylim([-0.02, 0.02])
    axs_velocity[0, 1].set_ylim([-0.2, 0.2])
    axs_velocity[1, 1].set_ylim([-0.2, 0.2])

    axs_velocity[0, 0].legend()
    axs_velocity[1, 0].legend()
    axs_velocity[0, 1].legend()
    axs_velocity[1, 1].legend()

    # Initialize acceleration plots
    fig_acceleration, axs_acceleration = plt.subplots(2, 2, figsize=(10, 8))
    line_xDD, = axs_acceleration[0, 0].plot([], [], 'm-', label='a_x')
    line_xDD_model, = axs_acceleration[0, 0].plot([], [], 'c-', label='a_x computed')
    line_yDD, = axs_acceleration[1, 0].plot([], [], 'm-', label='a_y')
    line_yDD_model, = axs_acceleration[1, 0].plot([], [], 'c-', label='a_y computed')
    line_q12DD, = axs_acceleration[0, 1].plot([], [], 'm-', label='q12DD')
    line_q12DD_model, = axs_acceleration[0, 1].plot([], [], 'c-', label='q12DD computed')
    line_q22DD, = axs_acceleration[1, 1].plot([], [], 'm-', label='q22DD')
    line_q22DD_model, = axs_acceleration[1, 1].plot([], [], 'c-', label='q22DD computed')

    axs_acceleration[0, 0].set_title('Acceleration x [m/s²]')
    axs_acceleration[1, 0].set_title('Acceleration y [m/s²]')
    axs_acceleration[0, 1].set_title('Acceleration q12 [rad/s²]')
    axs_acceleration[1, 1].set_title('Acceleration q22 [rad/s²]')

    axs_acceleration[0, 0].set_ylim([-0.02, 0.02])
    axs_acceleration[1, 0].set_ylim([-0.02, 0.02])
    axs_acceleration[0, 1].set_ylim([-0.2, 0.2])
    axs_acceleration[1, 1].set_ylim([-0.2, 0.2])

    axs_acceleration[0, 0].legend()
    axs_acceleration[1, 0].legend()
    axs_acceleration[0, 1].legend()
    axs_acceleration[1, 1].legend()

    return {
        'fig_position': fig_position, 'axs_position': axs_position,
        'lines_position': [line_x, line_x_model, line_y, line_y_model, line_q12, line_q12_model, line_q22, line_q22_model],
        'fig_velocity': fig_velocity, 'axs_velocity': axs_velocity,
        'lines_velocity': [line_xD, line_xD_model, line_yD, line_yD_model, line_q12D, line_q12D_model, line_q22D, line_q22D_model],
        'fig_acceleration': fig_acceleration, 'axs_acceleration': axs_acceleration,
        'lines_acceleration': [line_xDD, line_xDD_model, line_yDD, line_yDD_model, line_q12DD, line_q12DD_model, line_q22DD, line_q22DD_model]
    }

def update_plots(plot_data, time, data):
    # Update position plots
    plot_data['lines_position'][0].set_data(time, data['position'][0])  # x
    plot_data['lines_position'][1].set_data(time, data['position'][2])  # x computed
    plot_data['lines_position'][2].set_data(time, data['position'][1])  # y
    plot_data['lines_position'][3].set_data(time, data['position'][3])  # y computed
    plot_data['lines_position'][4].set_data(time, data['position'][4])  # q12
    plot_data['lines_position'][5].set_data(time, data['position'][6])  # q12 computed
    plot_data['lines_position'][6].set_data(time, data['position'][5])  # q22
    plot_data['lines_position'][7].set_data(time, data['position'][7])  # q22 computed

    # Update velocity plots
    plot_data['lines_velocity'][0].set_data(time, data['velocity'][0])  # v_x
    plot_data['lines_velocity'][1].set_data(time, data['velocity'][2])  # v_x computed
    plot_data['lines_velocity'][2].set_data(time, data['velocity'][1])  # v_y
    plot_data['lines_velocity'][3].set_data(time, data['velocity'][3])  # v_y computed
    plot_data['lines_velocity'][4].set_data(time, data['velocity'][4])  # q12D
    plot_data['lines_velocity'][5].set_data(time, data['velocity'][6])  # q12D computed
    plot_data['lines_velocity'][6].set_data(time, data['velocity'][5])  # q22D
    plot_data['lines_velocity'][7].set_data(time, data['velocity'][7])  # q22D computed

    # Update acceleration plots
    plot_data['lines_acceleration'][0].set_data(time, data['acceleration'][0])  # a_x
    plot_data['lines_acceleration'][1].set_data(time, data['acceleration'][2])  # a_x computed
    plot_data['lines_acceleration'][2].set_data(time, data['acceleration'][1])  # a_y
    plot_data['lines_acceleration'][3].set_data(time, data['acceleration'][3])  # a_y computed
    plot_data['lines_acceleration'][4].set_data(time, data['acceleration'][4])  # q12DD
    plot_data['lines_acceleration'][5].set_data(time, data['acceleration'][6])  # q12DD computed
    plot_data['lines_acceleration'][6].set_data(time, data['acceleration'][5])  # q22DD
    plot_data['lines_acceleration'][7].set_data(time, data['acceleration'][7])  # q22DD computed

    # Rescale axes to fit the data
    for ax in plot_data['axs_position'].flatten():
        ax.relim()
        ax.autoscale_view()

    for ax in plot_data['axs_velocity'].flatten():
        ax.relim()
        ax.autoscale_view()

    for ax in plot_data['axs_acceleration'].flatten():
        ax.relim()
        ax.autoscale_view()

    # Redraw the plots
    plot_data['fig_position'].canvas.draw()
    plot_data['fig_velocity'].canvas.draw()
    plot_data['fig_acceleration'].canvas.draw()

    plt.pause(0.001)  # Allows for dynamic updating

def main(args=None):
    # Initialize and start the ROS2 robot interface
    rclpy.init(args=args)
    robot = Robot("biglide")  # Modify this to test the biglide
    start_robot(robot)

    # Initialize Matplotlib plots
    plot_data = init_plots()

    # Start oscillations
    robot.start_oscillate()

    time_data = []
    position_data = [[], [], [], [], [], [], [], []]  # For x, y, q12, q22 and their computed models
    velocity_data = [[], [], [], [], [], [], [], []]  # For xD, yD, q12D, q22D and their computed models
    acceleration_data = [[], [], [], [], [], [], [], []]  # For xDD, yDD, q12DD, q22DD and their computed models

    while True:  # Runs the simulation indefinitely
        try:
            if robot.data_updated():
                # Test of the Direct Geometric Model
                # Data from gazebo
                q11 = robot.active_left_joint.position
                q21 = robot.active_right_joint.position
                x = robot.end_effector.position_x
                y = robot.end_effector.position_y
                q12 = robot.passive_left_joint.position
                q22 = robot.passive_right_joint.position

                # DGM
                x_model, y_model = dgm(q11, q21, -1)
                q12_model, q22_model = dgm_passive(q11, q21, -1)

                # Test of the Direct Kinematic Model
                q11D = robot.active_left_joint.velocity
                q21D = robot.active_right_joint.velocity
                xD = robot.end_effector.velocity_x
                yD = robot.end_effector.velocity_y
                q12D = robot.passive_left_joint.velocity
                q22D = robot.passive_right_joint.velocity

                xD_model, yD_model = dkm(q11, q12_model, q21, q22_model, q11D, q21D)
                q12D_model, q22D_model = dkm_passive(q11, q12_model, q21, q22_model, q11D, q21D, xD_model, yD_model)

                # Test of the Direct Kinematic Model Second order
                q11DD = robot.active_left_joint.acceleration
                q21DD = robot.active_right_joint.acceleration
                xDD = robot.end_effector.acceleration_x
                yDD = robot.end_effector.acceleration_y
                q12DD = robot.passive_left_joint.acceleration
                q22DD = robot.passive_right_joint.acceleration

                xDD_model, yDD_model = dkm2(q11, q12_model, q21, q22_model, q11D, q12D_model, q21D, q22D_model, q11DD, q21DD)
                q12DD_model, q22DD_model = dkm2_passive(q11, q12_model, q21, q22_model, q11D, q12D_model, q21D, q22D_model, q11DD, q21DD, xDD_model, yDD_model)

                time = robot.get_time()
                
                # Append new data to lists for plotting
                time_data.append(time)
                position_data[0].append(x)
                position_data[1].append(y)
                position_data[2].append(x_model)
                position_data[3].append(y_model)
                position_data[4].append(q12)
                position_data[5].append(q22)
                position_data[6].append(q12_model)
                position_data[7].append(q22_model)

                velocity_data[0].append(xD)
                velocity_data[1].append(yD)
                velocity_data[2].append(xD_model)
                velocity_data[3].append(yD_model)
                velocity_data[4].append(q12D)
                velocity_data[5].append(q22D)
                velocity_data[6].append(q12D_model)
                velocity_data[7].append(q22D_model)

                acceleration_data[0].append(xDD)
                acceleration_data[1].append(yDD)
                acceleration_data[2].append(xDD_model)
                acceleration_data[3].append(yDD_model)
                acceleration_data[4].append(q12DD)
                acceleration_data[5].append(q22DD)
                acceleration_data[6].append(q12DD_model)
                acceleration_data[7].append(q22DD_model)

                # Update plots
                data = {
                    'position': position_data,
                    'velocity': velocity_data,
                    'acceleration': acceleration_data
                }
                update_plots(plot_data, time_data, data)

                # Continue oscillations
                robot.continue_oscillations()

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main(sys.argv)
