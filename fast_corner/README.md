*How this project came to be ...*
1. I had very limited exposure to classical CV; thought SLAM would be a good place to start.
2. Tried reading the oft-cited ORB-SLAM2 paper -- didn't understand much.
3. Regressed back to ORB-SLAM1 -- not too helpful either; realised the **ORB** is the feature extractor.
4. Went to the ORB paper -- *yeah, no*. Turns out that's derived from *TWO* other CV approaches: FAST and BRIEF.
5. We are exploring FAST.

FAST stands for Features from Accelerated Segment Test. This was proposed by [Rosten and Drummond in their 2006 paper](https://www.edwardrosten.com/work/rosten_2006_machine.pdf). The ideas they explore and propose are very simple -- just wanted to implement this and see how robust this is.

Might eventually make a series out of this: FAST, BRIEF, ORB, and SLAM.

