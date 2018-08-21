import tensorflow as tf
import numpy as np
from tf_grouping import query_ball_point, query_ball_point2, group_point
from scipy.spatial.distance import cdist

class GroupPointTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
      with tf.device('/gpu:0'):
          points = tf.constant(np.random.random((1,128,16)).astype('float32'))
          print(points)
          xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
          xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
          radius = 0.3
          nsample = 32
          idx, pts_cnt = query_ball_point(radius, nsample, xyz1, xyz2)
          grouped_points = group_point(points, idx)
          print(grouped_points)

          # with self.test_session():
          with tf.Session() as sess:
              print("---- Going to compute gradient error")
              err = tf.test.compute_gradient_error(points, (1,128,16), grouped_points, (1,8,32,16))
              print(err)
              self.assertLess(err, 1e-4)


class QueryBallPoint2Test(tf.test.TestCase):

    def test(self):

        nbatch = 1
        xyz1 = np.random.random((nbatch, 128, 3)).astype('float32')
        xyz2 = np.random.random((nbatch, 8, 3)).astype('float32')
        radii = np.random.uniform(low=0.2, high=0.4, size=(nbatch, 8)).astype('float32')

        print('---- Verifying QueryBallPoint2')
        with tf.device('/gpu:0'):
            xyz1_tensor = tf.constant(xyz1)
            xyz2_tensor = tf.constant(xyz2)
            radii_tensor = tf.constant(radii)
            nsample = 32

            idx_op, pts_cnt_op = query_ball_point2(radii_tensor, nsample, xyz1_tensor, xyz2_tensor)

            with tf.Session() as sess:
                idx, pts_cnt = sess.run([idx_op, pts_cnt_op])

            assert(np.max(idx < 128))
            assert (np.max(pts_cnt <= nsample))

            for i in range(nbatch):  # For each model in batch
                Y = cdist(xyz1[i, :], xyz2[i, :], 'euclidean')

                within_ball = Y < np.expand_dims(radii[i, :], axis=0)
                pts_cnt_gt = np.sum(within_ball, axis=0)

                assert(np.all(pts_cnt[i, :] == pts_cnt_gt))

                for j in range(xyz2.shape[1]):  # For each cluster
                    assert(set(idx[i,j,:]) == set(np.nonzero(within_ball[:, j])[0]))

        pass



if __name__=='__main__':
    tf.test.main()
