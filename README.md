# MovieTrailerML

In this project we try to identify a movies genre by its trailer. We have gathered 30GB worth of movie trailers from youtube along with each movies genre. This is a project done for the class Introduction to Deep Learning at Reykjavik University.

The data we collected on each trailer is every hundredth frame of each trailer, all of the frames were taken in the worst quality available on youtube. Examples of these images would be, 

![Alt text](./eximg1.jpg?raw=true "Title")
![Alt text](./eximg2.jpg?raw=true "Title")

Many of the movies in our dataset have multiple genres, in these cases we created multiple instances of each film each with a single genre. We do not know what impact this will have on the results but we are excited to see.

## Troubleshooting
If you get an error telling you you need to install pydot when trying to draw the model, try executing these commands in this order:
```sh
pip intall pydot
pip intall pydotplus
pip intall graphviz
```