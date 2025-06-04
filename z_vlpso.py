import os
import random
import uuid
import copy
import numpy as np
import soundfile as sf
# from FAKEBOB.kaldi.egs.wsj.s5.steps.diagnostic.analyze_phone_length_stats import length
from utils import levenshteinDistance, unique_wav_path
from CMUPhoneme.string_similarity import CMU_similarity
from ALINEPhoneme.string_dissimilarity import ALINE_dissimilarity
from NISQA.predict import NISQA_score
from synthesis import audio_synthesis
from google_ASR import google_ASR
from iflytek_ASR import iflytek_ASR
from baidu_ASR import baidu_ASR
# from sensevoice_ASR import sensevoice_ASR
from speaker_sv import speaker_verification_gmm, speaker_verification_iv
from speaker_csi import gmm_ubm_csi, iv_plda_csi
from speaker_osi import gmm_ubm_osi, iv_plda_osi

class VLPSO():

    def __init__(self, reference_audio, reference_text, target_model, target, size, div):
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.target_model = target_model
        self.target = target
        self.size = size
        self.maxlen = 640
        self.div = div
        self.threshold_range = (-10, 10)

    def _np_softmax(self, input):
        exp_input = np.exp(input)
        softmax = exp_input / np.sum(exp_input, axis=-1, keepdims=True)
        return softmax

    def _initialize(self):
        self.init_speed = 0.9
        self.population = {}
        self.best_local = {}
        self.bird_count = {}
        self.bird_exampler = {}
        for _ in range(self.size):
            position = self._np_softmax(np.random.randn(((_//int(self.size/self.div))+1) * (self.maxlen//self.div), 32)) * 1
            position = position.flatten()
            bird = str(uuid.uuid4())
            self.population[bird] = (position, np.full(position.shape, self.init_speed))
            self.best_local[bird] = (position, 0)
            self.best_len = ((_//int(self.size/self.div))+1) * (self.maxlen//self.div) * 32
            self.bird_exampler[bird] = [0] * ((_//int(self.size/self.div))+1) * (self.maxlen//self.div)
            self.bird_count[bird] = 0

    def _estimate_threshold(self, conf_score, is_accepted, epsilon=0.1):
        inf, sup = self.threshold_range
        theta = (inf + sup) / 2
        if is_accepted and conf_score < theta:
            sup = conf_score
        elif not is_accepted and conf_score > theta:
            inf = min(conf_score, sup)
        self.threshold_range = (inf, sup)
        if sup - inf < epsilon:
            return True, theta
        else:
            return False, theta

    def _calculate_fitness(self, epoch):
        self.population_fitness = {}
        for count, bird in enumerate(self.population):

            position = self.population[bird][0]
            l_emo_numpy = position.reshape(-1, 32)

            audio_numpy = audio_synthesis(l_emo_numpy, self.reference_audio, self.reference_text)
            tmp_audio_file = './SampleDir/synthesis.wav'

            audio_quality = NISQA_score(tmp_audio_file)

            if 'ASR' in self.target_model:

                if self.target_model == 'googleASR':
                    transcription = google_ASR(tmp_audio_file)

                if self.target_model == 'iflytekASR':
                    transcription = iflytek_ASR(tmp_audio_file)

                if self.target_model == 'baiduASR':
                    transcription = baidu_ASR(tmp_audio_file)

                # if self.target_model == 'sensevoiceASR':
                #     transcription = sensevoice_ASR(tmp_audio_file)

                transcriped_file_name = self.target_model + '_' + transcription + '.wav'
                transcriped_file_path = unique_wav_path(os.path.join('./SampleDir', transcriped_file_name))
                sf.write(transcriped_file_path, audio_numpy, 22050)

                if levenshteinDistance(transcription, self.target) < 4:
                    success_file_name = 'success_' + self.target_model + '_' + transcription + '_vlpso' +'.wav'
                    success_file_path = unique_wav_path(os.path.join('./SuccessDir', success_file_name))
                    sf.write(success_file_path, audio_numpy, 22050)
                    print(f'has success epoch: {epoch}')

                if transcription == 'NA':
                    fitness_levenshtein = 100
                    fitness_CMU = 0
                    fitness_ALINE = 10000
                else:
                    fitness_levenshtein = levenshteinDistance(transcription, self.target) / (
                                (len(transcription) + len(self.target)) / 2)
                    fitness_CMU = CMU_similarity(transcription, self.target)
                    fitness_ALINE = ALINE_dissimilarity(transcription, self.target)

                fitness = -10 * fitness_levenshtein + 0.1 * fitness_CMU - 0.0001 * fitness_ALINE + 0.05 * audio_quality
                print(transcription)
                print(f"[Individual {bird} Fitness: {fitness:.2f}]")
                print(f"[Individual {bird} Levenshtein: {-10 * fitness_levenshtein:.2f}]")
                print(f"[Individual {bird} CMU: {0.1 * fitness_CMU:.2f}]")
                print(f"[Individual {bird} ALINE: {-0.0001 * fitness_ALINE:.2f}]")
                print(f"[Individual {bird} NISQA: {0.05 * audio_quality:.2f}]")
                print('\n')

            elif 'SV' in self.target_model:
                benign_wavs_rootdir = 'FAKEBOB/data/test-set/'
                benign_wavs_dir = os.path.join(benign_wavs_rootdir, self.target)

                if self.target_model == 'gmmSV':
                    is_accepted, threshold, conf_score = speaker_verification_gmm(tmp_audio_file, self.target,
                                                                                  benign_wavs_dir)

                if self.target_model == 'ivectorSV':
                    is_accepted, threshold, conf_score = speaker_verification_iv(tmp_audio_file, self.target,
                                                                                 benign_wavs_dir)
                if is_accepted:
                    success_file_name = 'success_' + self.target_model + '_' + self.target + '_vlpso' + '.wav'
                    success_file_path = unique_wav_path(os.path.join('./SuccessDir', success_file_name))
                    sf.write(success_file_path, audio_numpy, 22050)
                    print(f"The recognition for the speaker {self.target} passed!")

                converged, theta = self._estimate_threshold(conf_score, is_accepted)

                print(f"Threshold is now estimated at {theta}, the actual threshold is {threshold} \n")
                fitness_adv = -(max(theta, conf_score) - conf_score)

                fitness = fitness_adv + 0.02 * audio_quality
                # fitness = fitness_adv
                with open(r"z_log/pso_fit.txt", "a", encoding="utf-8") as file:
                    file.write(str(fitness) + "\n")
                    file.close()

                print(f"[Individual {bird} Fitness: {fitness:.2f}]")
                print(f"[Individual {bird} conf_score: {conf_score:.2f}]")
                print(f"[Individual {bird} Adv: {fitness_adv:.2f}]")
                print(f"[Individual {bird} NISQA: {0.02 * audio_quality:.2f}]")
                print('\n')

            elif 'CSI' in self.target_model:

                if self.target_model == "ivectorCSI":
                    max_score, target_label_score, decision, results_dict = iv_plda_csi(tmp_audio_file, self.target)

                elif self.target_model == "gmmCSI":
                    max_score, target_label_score, decision, results_dict = gmm_ubm_csi(tmp_audio_file, self.target)

                print(f"The recognized speaker is {decision}, the attack success is {(decision == self.target)}")

                if decision == self.target:
                    success_file_name = 'success_' + self.target_model + '_' + self.target + '_vlpso' +'.wav'
                    success_file_path = unique_wav_path(os.path.join('./SuccessDir', success_file_name))
                    sf.write(success_file_path, audio_numpy, 22050)

                fitness_adv = target_label_score - max_score

                fitness = fitness_adv + 0.02 * audio_quality

                print(f"[Individual {bird} Fitness: {fitness:.2f}]")
                print(f"[Individual {bird} conf_score: {target_label_score:.2f}]")
                print(f"[Individual {bird} Adv: {fitness_adv:.2f}]")
                print(f"[Individual {bird} NISQA: {0.02 * audio_quality:.2f}]")
                print('\n')

            elif 'OSI' in self.target_model:

                benign_wavs_dir = 'FAKEBOB/data/z-norm'

                if self.target_model == "ivectorOSI":
                    max_score, max_score_label, target_label_score, min_threshold, decision, results_dict = iv_plda_osi(
                        tmp_audio_file, benign_wavs_dir, self.target)

                elif self.target_model == "gmmOSI":
                    max_score, max_score_label, target_label_score, min_threshold, decision, results_dict = gmm_ubm_osi(
                        tmp_audio_file, benign_wavs_dir, self.target)

                target_score = results_dict[self.target][0]
                target_threshold = results_dict[self.target][1]
                is_accepted = results_dict[self.target][-1]

                if is_accepted:
                    success_file_name = 'success_' + self.target_model + '_' + self.target + '_vlpso' +'.wav'
                    success_file_path = unique_wav_path(os.path.join('./SuccessDir', success_file_name))
                    sf.write(success_file_path, audio_numpy, 22050)
                    print(f"The recognition for the speaker {self.target} passed!")

                converged, theta = self._estimate_threshold(target_score, is_accepted)

                print(f"Threshold is now estimated at {theta}, the actual threshold is {target_threshold} \n")

                fitness_adv = -(max(theta, target_score) - target_score)

                fitness = fitness_adv + 0.02 * audio_quality

                print(f"[Individual {bird} Fitness: {fitness:.2f}]")
                print(f"[Individual {bird} conf_score: {target_label_score:.2f}]")
                print(f"[Individual {bird} Adv: {fitness_adv:.2f}]")
                print(f"[Individual {bird} NISQA: {0.02 * audio_quality:.2f}]")
                print('\n')

            else:
                print('Not found model')

            if count == 0 and epoch == 0:
                self.best_global = position
                self.best_global_fitness = fitness
            if fitness >= self.best_global_fitness:
                self.best_global = position
                self.best_global_fitness = fitness
                print(f'best_global: {bird}, changed')
            if fitness >= self.best_local[bird][1]:
                self.best_local[bird] = (position, fitness)
                print(f'best_local: {bird} changed')

            self.population_fitness[bird] = (position, fitness)

    def run(self, iterations):
        self._initialize()
        self.bird_last = {}
        for epoch in range(iterations):
            self._calculate_fitness(epoch)
            current_population = copy.deepcopy(self.population)
            sorted_population = sorted(self.population_fitness.items(), key=lambda x: x[1][1], reverse=True)
            self.pc = {item[0]: rank + 1 for rank, item in enumerate(sorted_population)}
            for bird in current_population:
                position = current_population[bird][0]
                speed = current_population[bird][1]
                fitness = self.population_fitness[bird][1]
                feature = int(position.shape[0] / 32)

                if self.bird_count[bird] == 0:
                    '''Original VLPSO rank'''
                    # sorted_population = sorted(self.population_fitness.items(), key=lambda x: x[1][1], reverse=True)
                    # self.pc = {item[0]: rank + 1 for rank, item in enumerate(sorted_population)}
                    # pci = 0.05 + 0.45 * (np.exp(10 * (self.pc[bird] - 1) / (self.size - 1)) / (np.exp(10) - 1))
                    '''fitness rank'''
                    # sorted_population = sorted(self.population_fitness.items(), key=lambda x: x[1][1], reverse=True)
                    # self.pc = {item[0]: item[1][1] for item in sorted_population}
                    pci = 0.05 + 0.45 * (np.exp(10 * (self.pc[bird] - 1) / (self.size - 1)) / (np.exp(10) - 1))
                    length = position.shape[0]
                    filtered_population = {id: (position, fitness) for id, (position, fitness) in
                                           self.population.items()
                                           if position.shape[0] >= length and id != bird}
                    sampled_population = random.sample(list(filtered_population.items()), k=2)
                    for _ in range(feature):
                        if random.random() < pci:
                            exampler_id, (exampler_position, exampler_fitness) = max(sampled_population, key=lambda x: x[1][1].max())
                            self.bird_exampler[bird][_] = exampler_id
                        else:
                            self.bird_exampler[bird][_] = bird
                    self.bird_last[bird] = fitness

                if self.bird_count[bird] == 3:
                    if fitness >= self.bird_last[bird]:
                        print(f'bird: {bird} continue this exampler')
                    else:
                        print(f'bird: {bird} change exampler')
                        self.bird_count[bird] = 0
                glo = False
                if self.best_global.shape[0] >= position.shape[0]:
                    glo = False
                for _ in range(feature):
                    start = _ * 32
                    end = (_ + 1) * 32
                    w = 0.9 - 0.5 * ((epoch+1) / iterations)
                    c1 = 1.49445
                    c2 = 1.49445
                    exampler_position = self.best_local[self.bird_exampler[bird][_]][0]
                    if glo:
                        speed[start:end] = (w * speed[start:end] + c1 * random.random() * (
                                exampler_position[start:end] - position[start:end]) + c2 *
                                            random.random() * (self.best_global[start:end] - position[start:end]))
                    else:
                        speed[start:end] = w * speed[start:end] + c1 * random.random() * (
                                exampler_position[start:end] - position[start:end])
                position += speed
                position = position.reshape(-1, 32)
                position = self._np_softmax(position) * 1
                position = position.flatten()
                self.population[bird] = (position, speed)
                self.bird_count[bird] = self.bird_count[bird] + 1

            if (epoch+1) % 5 == 0:
                pop_fit = []
                fit_ls = list(self.population_fitness.values())
                su = 0
                for i in range(len(fit_ls)):
                    if (i+1) % (self.size/self.div) == 0:
                        pop_fit.append(su/(self.size/self.div))
                    else:
                        su += fit_ls[i][1]
                new_index = int(pop_fit.index(max(pop_fit))*(self.size/self.div))
                new_len = fit_ls[new_index][0].shape[0]
                if new_len == self.maxlen:
                    print(f'maxlen: {self.maxlen} continue')
                else:
                    print(f'maxlen change to {new_len}')
                    self.maxlen = new_len
                    unit = new_len / self.div
                    current_population2 = copy.deepcopy(self.population)
                    count2 = 0
                    for bird in current_population2:
                        if count2 < new_index or count2 >= new_index + unit:
                            length = int(unit * ((count2 // (self.size/self.div)) + 1))
                            position = current_population2[bird][0][:length]
                            position = position.reshape(-1, 32)
                            position = self._np_softmax(position) * 1
                            position = position.flatten()
                            fitness = self.population[bird][1]
                            self.population[bird] = (position, fitness)
                        count2 += 1

            print(f'epoch: {epoch} finish')

        if 'ASR' in self.target_model:
            return self.best_global

        elif 'SV' in self.target_model:
            return self.best_global, self.threshold_range

        elif 'CSI' in self.target_model:
            return self.best_global

        elif 'OSI' in self.target_model:
            return self.best_global, self.threshold_range



if __name__ == "__main__":
    reference_audio = '/home/iwindy/PycharmProjects/SMACK/Original_TheyDenyTheyLied.wav'
    reference_text = "They deny they lied"
    # target_model can be 'googleASR' or 'iflytekASR' or 'gmmSV' or 'ivectorSV'
    target_model = 'iflytekASR'
    # target can be speaker id or the target transcription
    # target = "librispeech_p1089"
    target = "They did not know they lied"
    # Run a small number of iterations with a small population size
    population_size = 30
    iterations = 20
    div = 5

    ga = VLPSO(reference_audio, reference_text, target_model, target, population_size, div)
    ga._initialize()
    # Run the Genetic Algorithm
    ga.run(iterations)
    # print(ga.best_len)
    # for i in ga.population:
    #     print(ga.population[i][0].shape)