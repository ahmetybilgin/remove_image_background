def remove_dominant_color(cv_frame, dm_color, frame_width, frame_height):

    for y in range(frame_height):
        for x in range(frame_width):

            pixel = cv_frame[y, x]
            if pixel[3] != 0 and pixel[0] == dm_color[0] and pixel[1] == dm_color[1] and pixel[2] == dm_color[2]:
                pixel[3] = 0

    return cv_frame


if __name__ == '__main__':

    remove_dominant_color(cv_frame=[], dm_color=[], frame_height=0, frame_width=0)
