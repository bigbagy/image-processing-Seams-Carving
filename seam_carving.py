import sys

import cv2

import numpy as np

import matplotlib.pyplot as plt

#replicate seam carving paper by Shai Avidan and Ariel Shamir

##########create energy function adn find cost matrix
class SeamCarving(object):
        
    def __init__(self, img):
        self.img = img

    def __abs_energy_function__(self, input_image):
        
        #First take the derivative of every image channel, then sum all color channels together, then compute and return cost matrix
        
        original_image = input_image
        # pad image edges using boarder_reflect101 method
        input_image = cv2.copyMakeBorder(input_image, 2, 2, 2, 2, borderType=cv2.BORDER_REFLECT101)
        energy_map = []
        for channel in range(input_image.ndim):
            dy, dx = np.gradient(input_image[:, :, channel].astype(float))
            dy=abs(dy)
            dx=abs(dx)
            energy_map = dy+ dx
        while energy_map.shape[0] != original_image.shape[0]:  # match rows
            energy_map = np.delete(energy_map, 0, axis=0)
        while energy_map.shape[1] != original_image.shape[1]:  # match columns
            energy_map = np.delete(energy_map, 0, axis=1)
        returned_cost_matrix = np.zeros(energy_map.shape)
        for row_n, row in enumerate(energy_map):
            for pixel_n, pixel in enumerate(row):
                if row_n == 0:  
                    returned_cost_matrix[row_n, pixel_n] = pixel

                if row_n != 0:   

                    if pixel_n == 0:   # left edge
                        seam_min = min(returned_cost_matrix[row_n - 1, pixel_n], returned_cost_matrix[row_n - 1, pixel_n + 1])
                        returned_cost_matrix[row_n, pixel_n] = pixel + seam_min


                    elif pixel_n == returned_cost_matrix.shape[1] - 1:  # right edge
                        seam_min = min(returned_cost_matrix[row_n - 1, pixel_n - 1],
                                       returned_cost_matrix[row_n - 1, pixel_n])
                        returned_cost_matrix[row_n, pixel_n] = pixel + seam_min


                    else:  # middle of image
                        returned_cost_matrix[row_n, pixel_n] = pixel + min(returned_cost_matrix[row_n - 1, pixel_n - 1],
                                                                           returned_cost_matrix[row_n - 1, pixel_n],
                                                                           returned_cost_matrix[row_n - 1, pixel_n + 1])
        return returned_cost_matrix


    def find_image_seam(self, energy_matrix, direction, return_matrix=False):
     ###Look at each pixel value and add it to the minimum of the three neighbors in a given direction.  

        print "optimal_seam in {} direction".format(direction)
        if direction == 'horizontal':
            rotated_matrix = energy_matrix.T
        elif direction == 'vertical':
            rotated_matrix = np.rot90(energy_matrix, 2)
        seam_matrix = np.zeros(rotated_matrix.shape)
        rows, cols = rotated_matrix.shape
        path = []
        for row_n, row in enumerate(rotated_matrix):  
            if row_n != 0:
                if min_pixel_ptr == 0:  # left edge
                    min_value = min(row[min_pixel_ptr + 1], row[min_pixel_ptr])
                    min_val_array = np.where(row == min_value)[0]
                    for i in range(len(min_val_array)):  
                        if np.allclose(min_val_array[i], min_pixel_ptr, atol=1) == True:
                            min_pixel_ptr = min_val_array[i]

                    row_n = abs(row_n - rows) - 1
                    seam_matrix[row_n, min_pixel_ptr] = min_value
                    if direction == 'horizontal':
                        path.append((min_pixel_ptr, row_n))
                    else:
                        path.append((row_n, min_pixel_ptr))
                elif min_pixel_ptr == cols - 1:  # right edge
                    min_value = min(row[min_pixel_ptr - 1], row[min_pixel_ptr])
                    min_val_array = np.where(row == min_value)[0]
                    for i in range(len(min_val_array)):  
                        if np.allclose(min_val_array[i], min_pixel_ptr, atol=1) == True:
                            min_pixel_ptr = min_val_array[i]
                    row_n = abs(row_n - rows) - 1
                    seam_matrix[row_n, min_pixel_ptr] = min_value
                    if direction == 'horizontal':
                        path.append((min_pixel_ptr, row_n))
                    else:
                        path.append((row_n, min_pixel_ptr))

                else:  #middle
                    min_value = min(row[min_pixel_ptr - 1], row[min_pixel_ptr], row[min_pixel_ptr + 1])
                    min_val_array = np.where(row == min_value)[0]
                    for i in range(len(min_val_array)):  
                        if np.allclose(min_val_array[i], min_pixel_ptr, atol=1) == True:
                            min_pixel_ptr = min_val_array[i]

                    row_n = abs(row_n - rows) - 1
                    seam_matrix[row_n, min_pixel_ptr] = min_value
                    if direction == 'vertical':
                        path.append((row_n, min_pixel_ptr))
                    else:
                        path.append((min_pixel_ptr, row_n))

            else:
                min_pixel_ptr = np.where(min(row) == row)[0][0]  # location of pixel

                row_n = abs(row_n - rows) - 1
                seam_matrix[row_n, min_pixel_ptr] = min(row)
                if direction != 'horizontal':
                    path.append((row_n, min_pixel_ptr))
                else:
                    path.append((min_pixel_ptr, row_n))


        if return_matrix == True and direction == 'horizontal':
            return seam_matrix.T, path
        if return_matrix == True and direction == 'vertical':
            return seam_matrix, path
        return path

##########delete seams
    def delete_seam(self, img, seam_path, direction):

        print 'deleting {} seam'.format(direction)
        if direction == 'vertical':
            new_image = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]))  # one vertical pixel per row
        elif direction == 'horizontal':
            new_image = np.zeros((img.shape[0] - 1, img.shape[1], img.shape[2]))  # one horizontal pixel per row

        else:
            return "wrong seam removal direction"

        rows, cols = new_image.shape[0], new_image.shape[1]

        horizontal_cols_edited = []
        vertical_rows_edited = []

        for row_n, row in enumerate(img):
            for pixel_n, pixel in enumerate(row):
                if (row_n, pixel_n) in seam_path:
                    for i in range(3):  
                        if direction == 'vertical':
                            vertical_rows_edited.append(row_n)
                            if pixel_n != cols:  
                                new_image[row_n, pixel_n, i] = img[row_n, pixel_n + 1, i]  
                            else:
                                pass
                        elif direction == 'horizontal':
                            horizontal_cols_edited.append(pixel_n)
                            if row_n == rows - 1:  
                                new_image[row_n, pixel_n, i] = img[row_n - 1, pixel_n, i]  
                            else:  
                                new_image[row_n, pixel_n, i] = img[row_n + 1, pixel_n, i]  

                else:
                    for i in range(3):  
                        if direction == 'horizontal':
                            if row_n == 0:
                                new_image[row_n, pixel_n, i] = img[row_n, pixel_n, i]
                            elif row_n == rows:
                                pass
                            else:
                                if pixel_n in horizontal_cols_edited:  
                                    new_image[row_n, pixel_n, i] = img[row_n + 1, pixel_n, i]
                                else:
                                    new_image[row_n, pixel_n, i] = img[row_n, pixel_n, i]
                        elif direction == 'vertical':
                            if pixel_n == cols:  
                                pass
                            else:
                                if row_n in vertical_rows_edited:
                                    new_image[row_n, pixel_n, i] = img[row_n, pixel_n + 1, i]
                                else:
                                    new_image[row_n, pixel_n, i] = img[row_n, pixel_n, i]
        return new_image




#########################################add seams
    def add_seam(self, img, seam_path, direction):

        print 'adding {} seam'.format(direction)
        if direction == 'vertical':
            new_image = np.zeros((img.shape[0], img.shape[1] + 1, img.shape[2]))  #  one vertical pixel per row
        if direction == 'horizontal':
            new_image = np.zeros((img.shape[0] + 1, img.shape[1], img.shape[2]))  # one horizontal pixel per row

        rows, cols = img.shape[0], img.shape[1]

        horizontal_cols_edited = []
        vertical_rows_edited = []

        for row_n, row in enumerate(new_image):
            for pixel_n, pixel in enumerate(row):
                if (row_n, pixel_n) in seam_path:
                    for i in range(3):  
                        if direction == 'horizontal':  # find average 

                            horizontal_cols_edited.append(pixel_n)
                            if row_n == 0: 
                                new_image[row_n + 1, pixel_n, i] = img[row_n, pixel_n, i]
                                new_image[row_n, pixel_n, i] = np.mean([img[row_n+1, pixel_n, i], img[row_n, pixel_n, i]])
                            elif (row_n == rows):  
                                new_image[row_n, pixel_n, i] = np.mean([img[row_n, pixel_n, i], img[row_n - 1, pixel_n, i]])
                                new_image[row_n + 1, pixel_n, i] = img[row_n, pixel_n, i]
                            else:
                                new_image[row_n, pixel_n, i] = img[row_n, pixel_n, i]
                                new_image[row_n + 1, pixel_n, i] = np.mean([img[row_n + 1, pixel_n, i], img[row_n - 1, pixel_n, i]])

                        if direction == 'vertical':  # average of the left/right neighbors

                            vertical_rows_edited.append(row_n)
                            if pixel_n == 0:  
                                new_image[row_n, pixel_n, i] = np.mean([img[row_n, pixel_n + 1, i], img[row_n, pixel_n, i]])
                                new_image[row_n, pixel_n + 1, i] = img[row_n, pixel_n, i]
                            elif (pixel_n == cols):  
                                new_image[row_n, pixel_n, i] = np.mean([img[row_n, pixel_n - 1, i], img[row_n, pixel_n, i]])
                                new_image[row_n, pixel_n + 1, i] = img[row_n, pixel_n, i]
                            else:
                                new_image[row_n, pixel_n, i] = img[row_n, pixel_n, i]
                                new_image[row_n, pixel_n + 1, i] = np.mean([img[row_n, pixel_n - 1, i], img[row_n, pixel_n + 1, i]])


                else:
                    for i in range(3):  
                        if direction == 'horizontal':

                            if row_n == rows :
                                new_image[row_n, pixel_n, i] = img[row_n - 1, pixel_n, i]
                            else:
                                if pixel_n in horizontal_cols_edited:
                                    new_image[row_n + 1, pixel_n, i] = img[row_n, pixel_n, i]
                                else:
                                    new_image[row_n, pixel_n, i] = img[row_n, pixel_n, i]



                        if direction == 'vertical':
                            if pixel_n == cols:  
                                new_image[row_n, pixel_n, i] = img[row_n, pixel_n - 1, i]
                            else:
                                if row_n in vertical_rows_edited:
                                    new_image[row_n, pixel_n + 1, i] = img[row_n, pixel_n, i]
                                else:
                                    new_image[row_n, pixel_n, i] = img[row_n, pixel_n, i]
        return new_image



####################################################
#control function remove seam
    def remove_seam_controller(self, number_horizontal_seams_to_remove=0, number_vertical_seams_to_remove=0):

        looped_image = self.img
        if (number_horizontal_seams_to_remove > 0) & (number_vertical_seams_to_remove > 0):
            seams = max(number_horizontal_seams_to_remove, number_vertical_seams_to_remove)
            for i in range(seams):
                if i % 2 == 0:
                    print "Finished ", i
                cost_map = self.__abs_energy_function__(looped_image)
                vertical_matrix, vertical_seam_location = self.find_image_seam(cost_map, 'vertical', True)
                horizontal_matrix, horizontal_seam_location = self.find_image_seam(cost_map, 'horizontal', True)
                looped_image = self.delete_seam(looped_image, vertical_seam_location, direction='vertical')
                looped_image = self.delete_seam(looped_image, horizontal_seam_location, direction='horizontal')

        elif number_horizontal_seams_to_remove > 0:
            for i in range(number_horizontal_seams_to_remove):
                if i % 2 == 0:
                    print "Finished ", i
                cost_map = self.__abs_energy_function__(looped_image)
                horizontal_matrix, horizontal_seam_location = self.find_image_seam(cost_map, 'horizontal', True)
                looped_image = self.delete_seam(looped_image, horizontal_seam_location, direction='horizontal')

        elif number_vertical_seams_to_remove > 0:
            seams = max(number_horizontal_seams_to_remove, number_vertical_seams_to_remove)
            for i in range(seams):
                if i % 2 == 0:
                    print "Finished ", i
                cost_map = self.__abs_energy_function__(looped_image)
                vertical_matrix, vertical_seam_location = self.find_image_seam(cost_map, 'vertical', True)
                looped_image = self.delete_seam(looped_image, vertical_seam_location, direction='vertical')

        return looped_image


########################################################
#control function add seam
    def add_seam_controller(self, number_horizontal_seams_to_add=0, number_vertical_seams_to_add=0):

        looped_image = self.img
        deleted_image = looped_image

        if (number_horizontal_seams_to_add > 0) & (number_vertical_seams_to_add > 0):
            seams = max(number_horizontal_seams_to_add, number_vertical_seams_to_add)
            for i in range(seams):

                cost_map = self.__abs_energy_function__(deleted_image)

                vert_matrix, vertical_seam_location = self.find_image_seam(cost_map, 'vertical', True)
                hor_matrix, horizontal_seam_location = self.find_image_seam(cost_map, 'horizontal', True)

                deleted_image = self.delete_seam(deleted_image, vertical_seam_location, direction='vertical')
                deleted_image = self.delete_seam(deleted_image, horizontal_seam_location, direction='horizontal')

                looped_image = self.add_seam(looped_image, vertical_seam_location, direction='vertical')
                looped_image = self.add_seam(looped_image, horizontal_seam_location, direction='horizontal')
        elif number_horizontal_seams_to_add > 0:
            for i in range(number_horizontal_seams_to_add):

                cost_map = self.__abs_energy_function__(deleted_image)
                hor_matrix, horizontal_seam_location = self.find_image_seam(cost_map, 'horizontal', True)
                deleted_image = self.delete_seam(deleted_image, horizontal_seam_location, direction='horizontal')
                looped_image = self.add_seam(looped_image, horizontal_seam_location, direction='horizontal')
        elif number_vertical_seams_to_add > 0:
            for i in range(number_horizontal_seams_to_add):

                cost_map = self.__abs_energy_function__(deleted_image)
                vert_matrix, vertical_seam_location = self.find_image_seam(cost_map, 'vertical', True)
                deleted_image = self.delete_seam(deleted_image, vertical_seam_location, direction='vertical')
                looped_image = self.add_seam(looped_image, vertical_seam_location, direction='vertical')
        return looped_image


#######################################
#control function optimal seam
    def optimal_seam_controller(self, number_horizontal_seams_to_remove=0, number_vertical_seams_to_remove=0):

        # using the optimal retargeting strategy
        looped_image = self.img

        original_rows, original_cols = looped_image.shape[0], looped_image.shape[1]
        transport_map = np.zeros((original_rows - number_horizontal_seams_to_remove, original_cols - number_vertical_seams_to_remove))
        print "Finding optimal_seam removal"

        for row_n, row in enumerate(transport_map):
            if row_n % 10 == 0:
                print row_n, 'row'
            for pixel_n, pixel in enumerate(row):
                if (row_n, pixel_n) == (0, 0):
                    pass
                else:
                    resized_horizontal = abs_energy_function(cv2.resize(looped_beach_image, (original_rows - row_n - 1, original_cols - pixel_n)))
                    horizontal_matrix, hort_path = find_image_seam(resized_horizontal, 'horizontal', return_matrix=True)
                    horizontal_cost = horizontal_matrix.sum()
                    resized_vertical = abs_energy_function(cv2.resize(looped_beach_image, (original_rows - row_n, original_cols - pixel_n - 1)))
                    vertical_matrix, vert_path = find_image_seam(resized_vertical, 'vertical', return_matrix=True)
                    vertical_cost = horizontal_matrix.sum()

                    if (transport_map[row_n - 1, pixel_n] + vertical_cost) < (transport_map[row_n, pixel_n - 1] + vertical_cost):
                        transport_map[row_n, pixel_n] = 1
                    else:
                        transport_map[row_n, pixel_n] = 2
        seam_retargeting = transport_map


        for row in np.rot90(seam_retargeting, 2):  # bottom to top
            for value in row:
                if value == 1:  # horizontal
                    cost_map = abs_energy_function(looped_image)
                    horizontal_matrix, horizontal_seam_location = find_image_seam(cost_map, 'horizontal', True)
                    looped_image = delete_seam(looped_image, horizontal_seam_location, direction='horizontal')
                elif value == 2:  # vertical
                    cost_map = abs_energy_function(looped_image)
                    vertical_matrix, vertical_seam_location = find_image_seam(cost_map, 'vertical', True)
                    looped_image = delete_seam(looped_image, vertical_seam_location, direction='vertical')
        return looped_image

#########################################################3
#main function
if __name__ == '__main__':
    img_name = sys.argv[1]
    add_or_remove = str(input("pls choose mode, type in number 1, 2 or 3 for add, remove or optimal"))
    # read in image
    read_image = cv2.imread("./images/{}".format(img_name))
    blue, green, red = np.rollaxis(read_image, 2)  # get correct channels
    og_image = np.dstack([red, green, blue])
    # create class instance
    img_seam = SeamCarving(og_image)

    # remove/add seams

    if add_or_remove == '1':
        h = raw_input("Add how many horizontal seams? (pls enter a number )")
        v = raw_input("Add how many vertical seams? (pls enter a number )")
        img_seam = SeamCarving(og_image)
        added_seams_img = img_seam.add_seam_controller(int(v), int(h))

    if add_or_remove == '2':
        h = raw_input("Remove how many horizontal ? (pls enter a number )")
        v = raw_input("Remove how many vertical? (pls enter a number )")
        removed_seams_img = img_seam.remove_seam_controller(int(v), int(h))

    if add_or_remove == '3':
        h = raw_input("Remove how many horizontal ? (pls enter a number )")
        v = raw_input("Remove how many vertical? (pls enter a number )")
        removed_seams_optimal_img = img_seam.remove_seam_controller(int(v), int(h))

######################################save output image
    if add_or_remove == '1':
        plt.imsave('addh_{}addv_{}seams_{}'.format(int(h), int(v), img_name), added_seams_img / 255)

    if add_or_remove == '2':
        plt.imsave('removeh_{}removev_{}seams_{}'.format(int(h), int(v), img_name), removed_seams_img / 255)

    if add_or_remove == '3':
        plt.imsave('removeh_{}removev_{}seams_optimal_{}'.format(int(h), int(v), img_name), removed_seams_optimal_img / 255)
