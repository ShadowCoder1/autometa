
7. Before reading any value, zoom in on the target itself: call `crop_image` on a tight box around
   the x position the target names, wide enough to hold both groups' symbols and the full height of
   their error bars. Read the datum and the caps from that magnified view, and only then map them
   back to the axis. Say in `notes` which zoomed box you read each group from.
