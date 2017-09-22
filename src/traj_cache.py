from trajopt_planner import *

if __name__ == '__main__':

	pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
	place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]

	pick_basic_EEtilt = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 200.0] 

	# initialize start/goal based on task 
	pick = pick_basic #pick_basic_EEtilt
	place = place_lower #place_lower 

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	weights = [0.0, 0.0, 0.0]
	T = 20.0

	featMethod = "ALL"
	numFeat = 1
	planner = Planner(2, False, featMethod, numFeat)

	traj_cache = {}

	for cup in [-1.0,0.0,1.0]:
		for table in [-1.0,0.0,1.0]:
			for laptop in [-10.0,0.0,10.0]:
				weights = [cup, table, laptop]
				traj = planner.replan(start, goal, weights, 0.0, T, 0.5)	

				if cup not in traj_cache:
					traj_cache[cup] = {}
				if table not in traj_cache[cup]:
					traj_cache[cup][table] = {}
				if laptop not in traj_cache[cup][table]:
					traj_cache[cup][table][laptop] = traj


	print traj_cache
	print "-------"
	print traj_cache[0.0][0.0][-10.0] 
