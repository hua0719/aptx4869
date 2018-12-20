import tensorflow as tf
from discriminator import *
from generator import *
from discriminator import *
from load_data import *
import time
import numpy as np
from utils import *
import scipy.misc


modelname= 'Auto2_eGAN_attenSegAN'
batch_size = 2
epoch = 20000
input_height = 224
input_width = 160
input_dim = 1
dataset_name = 'ISLES'
checkpoint_dir = 'checkpoint/'
result_dir = 'result/'
log_dir = 'log/'
numbatch = 150*28//batch_size

input_modalities = ['T1']
output_modalities = ['VFlair']



class edGAN(object):

    def __init__(self, sess, epoch, batch_size, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.num_batches = numbatch

        # train
        self.inputc = input_dim
        self.model_name = modelname
        self.learning_rate = 0.000001
        self.beta1 = 0.5

        # test
        self.sample_num = 16 # number of generated images to be saved

        # WGAN parameter
        self.disc_iters = 3  # The number of critic iterations for one-step of generator

        # auto parameter
        self.auto_t = 2

    def build_model(self):
        # some parameters
        image_dims = [input_height, input_width, input_dim]
        g_inputc = 1
        d_inputc = 1

        """ Graph Input """
        # image
        self.inputmod1 = tf.placeholder(tf.float32, [batch_size, input_height, input_width, g_inputc], name='input_mod1')
        self.inputmod2 = tf.placeholder(tf.float32, [batch_size, input_height, input_width, g_inputc], name='input_mod2')
        self.inputmod3 = tf.placeholder(tf.float32, [batch_size, input_height, input_width, g_inputc], name='input_mod3')
        self.targetmod = tf.placeholder(tf.float32, [batch_size, input_height, input_width, g_inputc], name='target_mod')
        # # noises
        # self.z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z')

        """ Loss Function """


        # marginal_likelihood = tf.reduce_sum(input * tf.log(G_decode ) + (1 - input) * tf.log(1 - G_decode),
        #                                     [1, 2])
        #
        # KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1,
        #                                     [1])
        # neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        # KL_divergence = tf.reduce_mean(KL_divergence)
        #
        # ELBO = -neg_loglikelihood - KL_divergence
        #
        # ED_LOSS = -ELBO

        """ Loss Function """

        # # sampling by re-parameterization technique
        # z = self.mu + sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        # output of D for real images
        D_real, D_real_logits, _, scale_real = d_model(x=self.targetmod, is_training=True, reuse=False)

        # output of D for fake imagesmn
        # encoding
        G_encode1 = encoder(self.inputmod1, modality='T1', is_training=True, reuse=False)             # 8*224*160*16
        G_encode2 = encoder(self.inputmod2, modality='T2', is_training=True, reuse=False)
        G_encode3 = encoder(self.inputmod3, modality='DWI', is_training=True, reuse=False)

        # 特征融合
        G_encode = tf.stack([G_encode1, G_encode2, G_encode3], axis=4)
        G_encode_max = tf.reduce_max(G_encode, axis=4)

        # decoding
        # 未融合
        G_out1 = decoder(G_encode1, is_training=True, reuse=False)
        G_out2 = decoder(G_encode2, is_training=True, reuse=True)
        G_out3 = decoder(G_encode3, is_training=True, reuse=True)
        # 融合后
        G_out = decoder(G_encode_max, is_training=True, reuse=True)

        D_fake, D_fake_logits, _, scale_fake = d_model(G_out, is_training=True, reuse=True)

        """ SegAN loss """
        self.scale_loss = tf.reduce_mean(tf.losses.absolute_difference(scale_real, scale_fake))

        self.d_loss = - self.scale_loss

        """ 融合损失 """
        # get loss for generator
        # 生成和目标间的mse
        self.mse_loss = tf.losses.mean_squared_error(self.targetmod, G_out)
        # 生成和目标间的MAE              论文c3
        self.MAE_loss = tf.reduce_mean(tf.losses.absolute_difference(self.targetmod, G_out))
        # 未融合特征生成图像与目标的损失  论文c1
        self.G_out1_loss = tf.reduce_mean(tf.losses.absolute_difference(self.targetmod, G_out1))
        self.G_out2_loss = tf.reduce_mean(tf.losses.absolute_difference(self.targetmod, G_out2))
        self.G_out3_loss = tf.reduce_mean(tf.losses.absolute_difference(self.targetmod, G_out3))
        self.c1_mae_loss = (self.G_out1_loss + self.G_out1_loss + self.G_out1_loss) / 3

        self.var_loss = var(G_encode)  # 论文 c2
        # self.g_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake))) + mse_loss

        self.g_loss = self.scale_loss + self.MAE_loss + self.var_loss + self.c1_mae_loss


        """ auto-context """
        # 把每轮输出拼接在输入上进行迭代，在每轮迭代时都进行优化
        a_G_out = G_out
        for t in range(1, self.auto_t+1):
            reuse = False
            if t > 1:
                reuse = True

            a_input1 = tf.concat([a_G_out, self.inputmod1], axis=3)
            a_G_encode1= encoder(a_input1, modality='T1_%s' % self.auto_t, is_training=True, reuse=reuse)
            a_input2 = tf.concat([a_G_out, self.inputmod2], axis=3)
            a_G_encode2 = encoder(a_input2, modality='T2_%s' % self.auto_t, is_training=True, reuse=reuse)
            a_input3 = tf.concat([a_G_out, self.inputmod3], axis=3)
            a_G_encode3 = encoder(a_input3, modality='DWI_%s' % self.auto_t, is_training=True, reuse=reuse)

            # 特征融合
            a_G_encode = tf.stack([a_G_encode1, a_G_encode2, a_G_encode3], axis=4)
            a_G_encode_max = tf.reduce_max(a_G_encode, axis=4)

            # decoding
            # 未融合的单个特征解码产生的结果  用来计算融合损失
            a_G_out1 = decoder(a_G_encode1, is_training=True, reuse=True)
            a_G_out2 = decoder(a_G_encode2, is_training=True, reuse=True)
            a_G_out3 = decoder(a_G_encode3, is_training=True, reuse=True)

            # 融合后

            a_G_out = decoder(a_G_encode_max, is_training=True, reuse=True)

            D_fake, D_fake_logits, _, a_scale_fake = d_model(a_G_out, is_training=True, reuse=True)

            """ SegAN loss """
            self.a_scale_loss = tf.reduce_mean(tf.losses.absolute_difference(scale_real, a_scale_fake))

            self.a_d_loss = - self.a_scale_loss

            # get loss for generator
            # 生成和目标间的mse
            self.a_mse_loss= tf.losses.mean_squared_error(self.targetmod, a_G_out)
            # 生成和目标间的MAE              论文c3
            self.a_MAE_loss = tf.reduce_mean(tf.losses.absolute_difference(self.targetmod, a_G_out))
            # 未融合特征生成的图像与目标的损失  论文c1
            self.a_G_out1_loss = tf.reduce_mean(tf.losses.absolute_difference(self.targetmod, a_G_out1))
            self.a_G_out2_loss = tf.reduce_mean(tf.losses.absolute_difference(self.targetmod, a_G_out2))
            self.a_G_out3_loss = tf.reduce_mean(tf.losses.absolute_difference(self.targetmod, a_G_out3))
            self.a_c1_mae_loss = (self.a_G_out1_loss+self.a_G_out2_loss+self.a_G_out3_loss)/3.0

            self.a_var_loss = var(a_G_encode)   #论文 c2
            # self.g_loss = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake))) + mse_loss

            self.a_g_loss = self.a_scale_loss + self.a_MAE_loss + self.a_var_loss + self.a_c1_mae_loss

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 2, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)
            """auto_context迭代的优化"""
            self.a_d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.a_d_loss, var_list=d_vars)
            self.a_g_optim = tf.train.AdamOptimizer(self.learning_rate * 2, beta1=self.beta1) \
                .minimize(self.a_g_loss, var_list=g_vars)

        # WGAN weight clipping
        # self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

        """" Testing """
        # for test
        la_code1 = encoder(self.inputmod1, modality='T1', is_training=False, reuse=True)
        la_code2 = encoder(self.inputmod2, modality='T2', is_training=False, reuse=True)
        la_code3 = encoder(self.inputmod3, modality='DWI',  is_training=False, reuse=True)

        la_encode = tf.stack([la_code1, la_code2, la_code3], axis=4)
        la_code_max = tf.reduce_max(la_encode, axis=4)

        self.fake_images = decoder(la_code_max, is_training=False, reuse=True)
        """auto_context的迭代"""
        self.t_fake_images = self.fake_images
        for t in range(1, self.auto_t+1):
            t_input1 = tf.concat([self.t_fake_images, self.inputmod1], axis=3)
            t_G_encode1 = encoder(t_input1, modality='T1_%s' % self.auto_t, is_training=False, reuse=True)
            t_input2 = tf.concat([self.t_fake_images, self.inputmod2], axis=3)
            t_G_encode2 = encoder(t_input2, modality='T2_%s' % self.auto_t, is_training=False, reuse=True)
            t_input3 = tf.concat([self.t_fake_images, self.inputmod3], axis=3)
            t_G_encode3 = encoder(t_input3, modality='DWI_%s' % self.auto_t, is_training=False, reuse=True)

            # 特征融合
            t_G_encode = tf.stack([t_G_encode1, t_G_encode2, t_G_encode3], axis=4)
            t_G_encode_max = tf.reduce_max(t_G_encode, axis=4)
            # 融合后
            self.t_fake_images = decoder(t_G_encode_max, is_training=False, reuse=True)



        """ Summary """
        # d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        # d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        MAE_loss = tf.summary.scalar("MAE_loss", self.MAE_loss)
        var_loss = tf.summary.scalar("var_loss", self.var_loss)
        c1_mae_loss = tf.summary.scalar("c1_mae_loss", self.c1_mae_loss)

        """ACM"""
        a_d_loss_sum = tf.summary.scalar("ACM_%d_d_loss" % t, self.a_d_loss)
        a_g_loss_sum = tf.summary.scalar("ACM_%d_g_loss" % t, self.a_g_loss)
        a_MAE_loss = tf.summary.scalar("ACM_%d_MAE_loss" % t, self.a_MAE_loss)
        a_var_loss = tf.summary.scalar("ACM_%d_var_loss" % t, self.a_var_loss)
        a_c1_mae_loss = tf.summary.scalar("ACM_%d_c1_mae_loss" % t, self.a_c1_mae_loss)



        # final summary operations
        self.g_sum = tf.summary.merge([g_loss_sum, MAE_loss, var_loss, c1_mae_loss])
        self.d_sum = tf.summary.merge([d_loss_sum])
        """ACM"""
        self.a_g_sum = tf.summary.merge([a_g_loss_sum,  a_MAE_loss,  a_var_loss, a_c1_mae_loss])
        self.a_d_sum = tf.summary.merge([a_d_loss_sum])


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # # graph inputs for visualize training results
        # self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        data_1 = np.load('ISLES/T1.npz')['arr_0']
        print(" Load T1...")
        data_2 = np.load('ISLES/T2.npz')['arr_0']
        print(" Load T2...")
        data_3 = np.load('ISLES/DWI.npz')['arr_0']
        print(" Load DWI...")
        data_4 = np.load('ISLES/VFlair.npz')['arr_0']
        print(" Load VFlair...")
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            self.input_modalities = input_modalities[0]
            self.output_modalities = output_modalities[0]


            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                i_batch_images_1 = loadnpz(data_1, idx*batch_size, batch_size)
                i_batch_images_2 = loadnpz(data_2, idx*batch_size, batch_size)
                i_batch_images_3 = loadnpz(data_3, idx*batch_size, batch_size)
                t_batch_images = loadnpz(data_4, idx*batch_size, batch_size)
                # batch_images = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                # batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                if could_load or (counter - 1) % self.disc_iters == 0:   #如果是加载，先更新一次D防止打印损失时没有值
                    _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum,
                                                            self.d_loss],
                                                           feed_dict={self.inputmod1: i_batch_images_1,
                                                                      self.inputmod2: i_batch_images_2,
                                                                      self.inputmod3: i_batch_images_3,
                                                                      self.targetmod: t_batch_images})
                    self.writer.add_summary(summary_str, counter)

                # update G network 
                _, summary_str, g_loss, MAE_loss, var_loss, c1_mae_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss,
                                                                                         self.MAE_loss, self.var_loss, self.c1_mae_loss],
                                                                                        feed_dict={self.inputmod1: i_batch_images_1,
                                                                                                   self.inputmod2: i_batch_images_2,
                                                                                                   self.inputmod3: i_batch_images_3,
                                                                                                   self.targetmod: t_batch_images})
                self.writer.add_summary(summary_str, counter)

                # display training status
                print("NO ACM iter")
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, mae_loss: %.8f, var_loss: %.8f, c1_mae_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, MAE_loss, var_loss, c1_mae_loss))
                """auto_context """
                if could_load or (counter - 1) % self.disc_iters == 0:  # 如果是加载，先更新一次D防止打印损失时没有值
                    _, summary_str, a_d_loss = self.sess.run([self.a_d_optim, self.a_d_sum,self.a_d_loss],
                                                           feed_dict={self.inputmod1: i_batch_images_1,
                                                                      self.inputmod2: i_batch_images_2,
                                                                      self.inputmod3: i_batch_images_3,
                                                                      self.targetmod: t_batch_images})
                    self.writer.add_summary(summary_str, counter)
                # update G network
                _, summary_str, a_g_loss, a_MAE_loss, a_var_loss, a_c1_mae_loss = self.sess.run(
                    [self.a_g_optim, self.a_g_sum, self.a_g_loss,self.a_MAE_loss, self.a_var_loss, self.a_c1_mae_loss],
                    feed_dict={self.inputmod1: i_batch_images_1,
                               self.inputmod2: i_batch_images_2,
                               self.inputmod3: i_batch_images_3,
                               self.targetmod: t_batch_images})
                self.writer.add_summary(summary_str, counter)

                # display training status
                print("after ACM iter" )
                print(
                    "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, mae_loss: %.8f, var_loss: %.8f, c1_mae_loss: %.8f" \
                    % (
                    epoch, idx, self.num_batches, time.time() - start_time, a_d_loss, a_g_loss, a_MAE_loss, a_var_loss,
                    a_c1_mae_loss))


                counter += 1
                # # save training results for every 300 steps
                # if np.mod(counter, 300) == 0:
                #     sample_in = i_batch_images
                #     sample_target = t_batch_images
                #     if np.max(i_batch_images) == 0:                  #如果当前batch恰巧全为背景
                #         sample_in = loadnpz(data_1, (idx-2)*batch_size, batch_size)
                #         sample_target =loadnpz(data_2, (idx-2)*batch_size, batch_size)
                #
                #
                #     input_t = 255 / np.max(sample_in)
                #     input_sample_slice = sample_in[4, :, :, 0] * input_t
                #     scipy.misc.imsave('./' + check_folder(
                #         self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                #                       '_train_{:02d}_{:04d}_input.png'.format(epoch, idx), input_sample_slice)
                #
                #
                #     samples_gen = self.sess.run(self.fake_images, feed_dict={self.inputmod: sample_in})
                #     gen_t = 255 / np.max(samples_gen)
                #     gen_samples_slice = samples_gen[4, :, :, 0]*gen_t
                #     scipy.misc.imsave('./' + check_folder(
                #         self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                #                       '_train_{:02d}_{:04d}_generate.png'.format(
                #         epoch, idx), gen_samples_slice)
                #
                #     target_t = 255 / np.max(sample_target)
                #     target_samples_slice = t_batch_images[4, :, :, 0] * target_t
                #     scipy.misc.imsave('./' + check_folder(
                #         self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                #                       '_train_{:02d}_{:04d}_target.png'.format(
                #                           epoch, idx), target_samples_slice)
                #
                #
                #     self.save(self.checkpoint_dir, counter)


                    # tot_num_samples = min(self.sample_num, self.ba0tch_size)
                    # # manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    # # manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    # # save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                    # #             './' + check_folder(
                    # #                 self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                    # #                 epoch, idx))

            # After an epoch, save training results

            r = random.randint(0, 27)
            sample_in_1 = data_1[r, 70:70+batch_size, 0:-6, 34:-36, np.newaxis]
            sample_in_2 = data_2[r, 70:70+batch_size, 0:-6, 34:-36, np.newaxis]
            sample_in_3 = data_3[r, 70:70+batch_size, 0:-6, 34:-36, np.newaxis]
            sample_target = data_4[r, 70:70+batch_size, 0:-6, 34:-36, np.newaxis]

            input_t_1 = 255 / np.max(sample_in_1)
            input_sample_slice = sample_in_1[0, :, :, 0] * input_t_1
            scipy.misc.imsave('./' + check_folder(
                self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                              '_train_{:02d}_{:04d}_input_T1.png'.format(epoch, r), input_sample_slice)
            input_t_2 = 255 / np.max(sample_in_2)
            input_sample_slice = sample_in_2[0, :, :, 0] * input_t_2
            scipy.misc.imsave('./' + check_folder(
                self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                              '_train_{:02d}_{:04d}_input_T2.png'.format(epoch, r), input_sample_slice)
            input_t_3 = 255 / np.max(sample_in_3)
            input_sample_slice = sample_in_3[0, :, :, 0] * input_t_3
            scipy.misc.imsave('./' + check_folder(
                self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                              '_train_{:02d}_{:04d}_input_DWI.png'.format(epoch, r), input_sample_slice)

            samples_gen = self.sess.run(self.fake_images, feed_dict={self.inputmod1: sample_in_1,
                                                                     self.inputmod2: sample_in_2,
                                                                     self.inputmod3: sample_in_3,})
            gen_t = 255 / np.max(samples_gen)
            gen_samples_slice = samples_gen[0, :, :, 0]*gen_t



            scipy.misc.imsave('./' + check_folder(
                self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                              '_train_{:02d}_{:04d}_generate.png'.format(epoch, r), gen_samples_slice)
            """ACM"""
            samples_gen = self.sess.run(self.t_fake_images, feed_dict={self.inputmod1: sample_in_1,
                                                                     self.inputmod2: sample_in_2,
                                                                     self.inputmod3: sample_in_3, })
            gen_t = 255 / np.max(samples_gen)
            gen_samples_slice = samples_gen[0, :, :, 0] * gen_t
            scipy.misc.imsave('./' + check_folder(
                self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                              '_train_{:02d}_{:04d}_Acm_generate.png'.format(epoch, r), gen_samples_slice)


            target_t = 255 / np.max(sample_target)
            target_samples_slice = sample_target[0, :, :, 0] * target_t
            scipy.misc.imsave('./' + check_folder(
                self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                              '_train_{:02d}_{:04d}_target.png'.format(epoch, r), target_samples_slice)


            self.save(self.checkpoint_dir, counter)
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            #self.save(self.checkpoint_dir, counter)

            # show temporal results
#            self.visualize_results(epoch)

            # save model for final step
            #self.save(self.checkpoint_dir, counter)





    # def visualize_results(self, epoch):
    #     tot_num_samples = min(self.sample_num, self.batch_size)
    #     image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
    #
    #     """ random condition, random noise """
    #
    #     z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
    #
    #     samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})
    #
    #     save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
    #                 check_folder(
    #                     self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.inputc)


    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)


    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0



def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        egan = edGAN(sess, epoch=epoch,batch_size=batch_size,dataset_name=dataset_name,checkpoint_dir=checkpoint_dir,
                     result_dir=result_dir,
                     log_dir=log_dir)
        egan.build_model()
        show_all_variables()

        # launch the graph in a session
        egan.train()
        print(" [*] Training finished!")

if __name__ == '__main__':
    main()

