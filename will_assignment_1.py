#Brian Will 
#DSCI 401 
#assignment 1

#1 flatten function 
#Takes a list of Elements and makes it into a flat list. 
def flatten(List):
	x = List
	flat_list = []
	if hasattr(x, "__iter__"):
		for element in x: 
			if hasattr(element, "__iter__"):
				flat_list.extend(flatten(element))
			else:
				flat_list.append(element)
		return flat_list
	else: 
		flat_list = [x]
		return flat_list
 
 #2 power set
 #finds a power set of a list    	
def powerset(List):
	x = List
	power_set_list = []
	if hasattr(x, "__iter__"):
		if x:
			split = powerset(x[1:])
			temp = [] 
			for element in split: 
				#print(element)
				temp.append(element + x[:1])
			power_set_list = split+temp
			return power_set_list
		else:
   			return [[]]
	else: 
   		return("needs valid input")
   		
#3 all_perms 
#finds permutations of a list of numbers 
def all_perms(list):

	if list == []: 
		return []
	if len(list) == 1:
		return(list)
		
	perm = []
	set_add = [list[-1]]
	set_add1 = [list[-2]]
	x=len(list)
	set_add2 = list[:(x-2)]
	
	for i in range(len(set_add) + 1):
		perm.append(set_add[:i] + set_add1 + set_add[i:])
	
	for rem in set_add2:
		count = 0
		#print("starting_loop rem is: " + str(rem))
		perm_1 = perm
		perm = []
		while(count <= len(perm_1[1])): 
			for i in perm_1:
				thing_to_add = i[:count] +[rem] + i[count:]
				perm.append(thing_to_add)
			count = count + 1  
      	#print(count)
      	#print("break")
	return perm

#spiral problem. 
#always starts in corner one no matter what corner you feed it. 
#I made functions to move up down left and right
#but did not have time to make one for starting in each corner. 
#I am confident given another day, had I had better time management 
#I would have done it. 
def spiral(x,y):
	sorry = y
	count_1 = 0
	count_2 = 0
	matrix = []
	#Make the Matrix  
	while count_1 < x:
		row = [] 
		matrix.append(row)
		while count_2 < x: 
			matrix[count_1].append(-1)
			count_2 = count_2 +1
		count_2 = 0
		count_1 = count_1 + 1
  
	result_count = (x*x) - 1
	#Helper functions to move around the matrix 
	def move_right(mx, bound, place, rc):
		count = 0
		while count < bound:
			if(mx[place[0]][place[1]] == -1):
				mx[place[0]][place[1]] = rc
				rc = rc - 1
				place[1] = place[1] + 1
			else: 
				place[1] = place[1] + 1
			count = count + 1
		return rc
	
	def move_down(mx, bound, place, rc):
		count = 0
		while count < bound:
			if(mx[place[0]][place[1]] == -1):
				mx[place[0]][place[1]] = rc
				rc = rc - 1
				place[0] = place[0] + 1
			else: 
				place[0] = place[0] + 1
			count = count + 1
			#for i in mx: 
				#print(i)
		return rc

	def move_left(mx, bound, place, rc):
		count = 0
		while count < bound:
			if(mx[place[0]][place[1]] == -1):
				mx[place[0]][place[1]] = rc
				rc = rc - 1
				place[1] = place[1] - 1
			else: 
				place[1] = place[1] - 1
			count = count + 1
			#for i in mx: 
				#print(i)
		return rc
		
	def move_up(mx, bound, place, rc):
		count = 0
		while count < bound:
			if(mx[place[0]][place[1]] == -1):
				mx[place[0]][place[1]] = rc
				rc = rc - 1
				place[0] = place[0] - 1
			else: 
				place[0] = place[0] - 1
			count = count + 1
		return rc
	
	#spiral starting in corner one
	start = [0,0]
	special = 0
	while result_count >= 0: 
		start[0] = special
		start[1] = special
		#print(start)
		result_count = move_right(matrix, x, start, result_count)
		start[0] = special
		start[1] = x-1
		#print(start)
		result_count = move_down(matrix, x, start, result_count)
		start[0] = x-1
		start[1] = x-1
		#print(start)
		result_count = move_left(matrix, x, start, result_count)
		start[0] = x-1
		start[1] = special
		#print(start)
		result_count = move_up(matrix, x, start, result_count)
		x = x - 1
		special = special + 1

	for i in matrix: 
		print(i)		




