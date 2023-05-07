import random
import numpy as np
import scipy.spatial.distance as spd

def overlap(locations, next_loc, n):
    #Check whether pellets are disjoint
    next_loc = np.asarray(next_loc)
    max_dist = np.sqrt(2)*n
    for loc in locations:
        loc = np.asarray(loc)
        dist = np.linalg.norm(loc-next_loc)
        if (dist<max_dist):
            return True
    return False

def valid_indicies(locations, pellet_size):
    max_dist = np.sqrt(2)*pellet_size
    distances = spd.squareform(spd.pdist(locations))
    idx = []
    for i in range(len(locations)):
        for j in range(i+1,len(locations)):
            if(distances[i,j]<=max_dist):
                idx.append(i)
                break
    return idx

def pellet_location(n, num_pellets, pellet_size):
    # n is the size of the image nxn
    # num_pellets is the number of pellets in log

    # Generate a random number coordinates for lead and steel pellets inside the logs circle
    # Returns a list with locations of pellets

    locations = []
    
    for i in range(num_pellets):
        radius = n//2 - pellet_size #Ensures that the pellet does cross the boundary of the log
        c1,c2 = (n//2, n//2)
        theta = 2*np.pi*random.uniform(0,1)
        r = radius * random.uniform(0,1)

        x_pellet = int(c1 + r*np.cos(theta))
        y_pellet = int(c2 + r*np.sin(theta))
        locations.append((x_pellet,y_pellet))
    return locations

def initialize_im(n,att_coef_wood):
    # n: size of the image nxn
    im = np.zeros((n,n))
    c1,c2 = (n//2, n//2)
    radius = n//2
    for i in range(n):
        for j in range(n):
            if ((i - c1)**2 + (j - c2)**2) <= radius**2:
                im[i][j] = att_coef_wood
    return im

def add_pellet(image, loc, pellet_size, att_coef):
    # Loc: (i,j) center of pellet
    x_center, y_center = loc[0],loc[1]
    radius = pellet_size//2

    #Insert pellet
    image[np.abs(x_center-radius):x_center+radius+1,np.abs(y_center-radius):y_center+radius+1] = att_coef
    return image

def generate_im(n,att_coefs,num_pellets,pellet_size):
    att_wood = att_coefs[1]
    im = initialize_im(n,att_wood)
    locations = pellet_location(n,num_pellets,pellet_size)
    idx = valid_indicies(locations,pellet_size)
    num_steel = np.random.randint(0,len(locations))

    for i in set(range(num_pellets))-set(idx):
        loc = locations[i]
        if (i<num_steel):
            im = add_pellet(im,loc, pellet_size, att_coefs[1])
        else:
            im = add_pellet(im,loc, pellet_size, att_coefs[2])
    return im



