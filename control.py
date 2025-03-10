from lab_amoro.parallel_robot import *
from lab_amoro.plot_tools import *
from biglide_models import *  # Modify this to use the biglide
import sys
import matplotlib.pyplot as plt

assembly_mode = -1
gamma1 = gamma2 = -1
Kp = 60
Kd = 15

def main(args=None):
	# Initialize and start the ROS2 robot interface
	rclpy.init(args=args)
	robot = Robot("biglide")  # Modify this to use the biglide
	start_robot(robot)
	
	# Create real-time plot windows
	fig, axes = plt.subplots(2, 1, figsize=(8, 10))
	time_data = []
	joint1_actual_data = []
	joint2_actual_data = []
	joint1_desired_data = []
	joint2_desired_data = []

	ln1, = axes[0].plot([], [], 'm-', label="Joint 1 Actual")
	ln1_desired, = axes[0].plot([], [], 'k-.', label="Joint 1 desired position")
	ln2, = axes[1].plot([], [], 'm-', label="Joint 2 Actual")
	ln2_desired, = axes[1].plot([], [], 'k-.', label="Joint 2 desired position")

	axes[0].set_title("Joint 1 position [m]")
	axes[0].set_xlabel("Time [s]")
	axes[0].set_ylabel("Position [rad]")
	axes[0].legend()

	axes[1].set_title("Joint 2 position [m]")
	axes[1].set_xlabel("Time [s]")
	axes[1].set_ylabel("Position [rad]")
	axes[1].legend()
	# Prepare plots (scopes)
	app = QtGui.QApplication([])
	scope_joint1 = Scope("Joint 1", -0.5, 6)
	scope_joint2 = Scope("Joint 2", -1.5, 6)

	# Create the trajectory as arrays in CARTESIAN SPACE (position, velocity, acceleration)
	tf = 2
	q11 = robot.active_left_joint.position
	q21 = robot.active_right_joint.position
	xi, yi = dgm(q11,q21,-1)
	xf = xi + 0.06
	yf = yi + 5
	t = np.linspace(0,2,200) # divide the interval of 2s in 200 instants, since we want a period of 10ms
	
	s = 10 * np.power(t/tf,3) - 15 * np.power(t/tf,4) + 6 * np.power(t/tf,5)
	x = xi + (xf - xi) * s
	y = yi + (yf - yi) * s
	
	v = 6*5/tf * np.power(t/tf,4) - 15*4/tf * np.power(t/tf,3) + 10*3/tf * np.power(t/tf,2)
	vx = (xf - xi) * v
	vy = (yf - yi) * v
	a = 6*5*4/(tf*tf) * np.power(t/tf,3) - 15*4*3/(tf*tf) * np.power(t/tf,2) + 10*3*2/(tf*tf) * (t/tf)
	ax = (xf - xi) * a
	ay = (yf - yi) * a

	n_samples = len(x)
	# Create the trajectory as arrays in JOINT SPACE using the inverse models (position, velocity, acceleration)
	q11s = np.zeros(n_samples)
	q21s = np.zeros(n_samples)
	q12s = np.zeros(n_samples)
	q22s = np.zeros(n_samples)
	q11Ds = np.zeros(n_samples)
	q21Ds = np.zeros(n_samples)
	q12Ds = np.zeros(n_samples)
	q22Ds = np.zeros(n_samples)
	q11DDs = np.zeros(n_samples)
	q21DDs = np.zeros(n_samples)
	q12DDs = np.zeros(n_samples)
	q22DDs = np.zeros(n_samples)
	for i in range(n_samples):
		q11s[i], q21s[i] = igm(x[i], y[i], gamma1, gamma2)
		q12s[i], q22s[i] = dgm_passive(q11s[i], q21s[i], assembly_mode)
		q11Ds[i], q21Ds[i] = ikm(q11s[i], q12s[i], q21s[i], q22s[i], vx[i], vy[i])
		q12Ds[i], q22Ds[i] = dkm_passive(q11s[i], q12s[i], q21s[i], q22s[i], q11Ds[i], q21Ds[i], vx[i], vy[i])
		q11DDs[i], q21DDs[i] = ikm2(q11s[i], q12s[i], q21s[i], q22s[i], q11Ds[i], q12Ds[i], q21Ds[i], q22Ds[i], ax[i], ay[i])
		q12DDs[i], q22DDs[i] = dkm2_passive(q11s[i], q12s[i], q21s[i], q22s[i], q11Ds[i], q12Ds[i], q21Ds[i], q22Ds[i], q11DDs[i], q21DDs[i], ax[i], ay[i])
	
	index = 0
	error_x = []
	error_y = []
	#tau = np.zeros(2,n_samples)
	# Controller
	try:
		robot.apply_efforts(0.0, 0.0)  # Required to start the simulation
		while True:
			if robot.data_updated():
				# Robot available data - This is the only data thet you can get from a real robot (joint encoders)
				q11 = robot.active_left_joint.position
				q21 = robot.active_right_joint.position
				q11D = robot.active_left_joint.velocity
				q21D = robot.active_right_joint.velocity
				
				q12, q22 = dgm_passive(q11, q21, assembly_mode)
				q12D, q22D = dkm_passive(q11, q12, q21, q22, q11D, q21D, vx[index], vy[index])
				# CTC controller
				M,C = dynamic_model(q11, q12, q21, q22, q11D, q12D, q21D, q22D)
				c = np.array(C[0][0], C[1][0])
				qtDD = np.array([q11DDs[index],q21DDs[index]])
				qtD = np.array([q11Ds[index],q21Ds[index]])
				qt = np.array([q11s[index],q21s[index]])
				qaD = np.array([q11D, q21D])
				qa = np.array([q11, q21])

				tau = M @ (qtDD + Kd*(qtD-qaD) + Kp*(qt-qa)) + c
				tau_left = tau[0]
				tau_right = tau[1]
				#robot.apply_efforts(0.0, 0.0)
				robot.apply_efforts(tau_left, tau_right)

				# Scope update
				time = robot.get_time()
				if time < 2.5:
					time_data.append(time)
					joint1_actual_data.append(q11)
					joint2_actual_data.append(q21)
					joint1_desired_data.append(q11s[index])
					joint2_desired_data.append(q21s[index])
					error_x.append(q11s[index] - q11)
					error_y.append(q21s[index] - q21)

					# Update plot
					ln1.set_data(time_data, joint1_actual_data)
					ln1_desired.set_data(time_data, joint1_desired_data)
					ln2.set_data(time_data, joint2_actual_data)
					ln2_desired.set_data(time_data, joint2_desired_data)
					
					axes[0].relim()
					axes[0].autoscale_view()
					axes[1].relim()
					axes[1].autoscale_view()

					plt.pause(0.01)

				if index < len(x) - 1:
					index += 1  # Next point in trajectory

	except KeyboardInterrupt:
		
		print("mean of q11= ", np.mean(error_x))
		print("max of q11=", np.max(error_x))
		print("mean of q21= ", np.mean(error_y))
		print("max of q21 = ", np.max(error_y))

		plt.figure()
		plt.plot(error_x)
		plt.title("Error of q11")
		plt.xlabel("Time")
		plt.ylabel("Error")
		plt.grid()

		plt.figure()
		plt.plot(error_y)
		plt.title("Error of q21")
		plt.xlabel("Time [ms]")
		plt.ylabel("Error [m]")
		plt.grid()

		plt.show()

	finally:
		plt.close('all')
		rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
