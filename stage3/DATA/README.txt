songs.csv contains the original Song table. It has 961,593 tuples.

tracks.csv contains the original Track table. It has 734,485 tuples.

sample_A.csv contains the downsampled Song table, with 4,192 tuples. It has the
same schema as songs.csv.

sample_B.csv contains the downsampled Track table, with 5,000 tuples. It has
the same schema is tracks.csv.

C.csv contains the survivors of blocking, with 5,223 tuples. The first column
is a unique id. The next two columns are ltable_id (the id of the song) and
rtable_id (the id of the track). The remaining columns are the non-id columns
from sample_A.csv, followed by the non-id columns from sample_B.csv.

G.csv contains the sampled and labeled candidates from C.csv. It has 500
candidate tuples from C.csv. The schema is the same as that of C.csv, with an
additional column gold_label, where the value can be 0 (not a true match) or 1
(true match).

I.csv contains the training set. It has 350 candidate tuples from G.csv, and
the same schema as G.csv.

J.csv contains the testing set. It has the remaining 150 candidate tuples from
G.csv, and the same schema as G.csv.
