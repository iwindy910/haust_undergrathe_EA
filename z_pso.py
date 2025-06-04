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
# from SenseVoice.sensevoice_ASR import sensevoice_ASR
from speaker_sv import speaker_verification_gmm, speaker_verification_iv
# from speaker_csi import gmm_ubm_csi, iv_plda_csi
# from speaker_osi import gmm_ubm_osi, iv_plda_osi

class PSO():

    def __init__(self, reference_audio, reference_text, target_model, target, size):
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.target_model = target_model
        self.target = target
        self.size = size
        self.threshold_range = (-10, 10)

    def _np_softmax(self, input):
        exp_input = np.exp(input)
        softmax = exp_input / np.sum(exp_input, axis=-1, keepdims=True)
        return softmax

    def _initialize(self):
        self.length = 330
        self.init_speed = 0.9
        self.population = {}
        self.best_local = {}
        for _ in range(self.size):
            position = self._np_softmax(np.random.randn(self.length, 32)) * 0.25
            position = position.flatten()
            bird = str(uuid.uuid4())
            self.population[bird] = (position, self.init_speed)
            self.best_local[bird] = (position, 0)

    def _estimate_threshold(self, conf_score, is_accepted, epsilon=0.1):
        inf, sup = self.threshold_range
        theta = (inf + sup) / 2
        if is_accepted and conf_score < theta:
            sup = conf_score
        elif not is_accepted and conf_score > theta:
            inf = conf_score
        self.threshold_range = (inf, sup)
        if sup - inf < epsilon:
            return True, theta
        else:
            return False, theta

    def _calculate_fitness(self, epoch):
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
                    success_file_name = 'success_' + self.target_model + '_' + transcription + '.wav'
                    success_file_path = unique_wav_path(os.path.join('./SuccessDir', success_file_name))
                    sf.write(success_file_path, audio_numpy, 22050)

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

                print(f"[Individual {bird} Fitness: {fitness:.2f}]")
                print(f"[Individual {bird} Levenshtein: {-10 * fitness_levenshtein:.2f}]")
                print(f"[Individual {bird} CMU: {0.1 * fitness_CMU:.2f}]")
                print(f"[Individual {bird} ALINE: {-0.0001 * fitness_ALINE:.2f}]")
                print(f"[Individual {bird} NISQA: {0.05 * audio_quality:.2f}]")
                print('\n')

            if 'SV' in self.target_model:
                benign_wavs_rootdir = 'FAKEBOB/data/test-set/'
                benign_wavs_dir = os.path.join(benign_wavs_rootdir, self.target)

                if self.target_model == 'gmmSV':
                    is_accepted, threshold, conf_score = speaker_verification_gmm(tmp_audio_file, self.target,
                                                                                  benign_wavs_dir)

                if self.target_model == 'ivectorSV':
                    is_accepted, threshold, conf_score = speaker_verification_iv(tmp_audio_file, self.target,
                                                                                 benign_wavs_dir)
                if is_accepted:
                    success_file_name = 'success_' + self.target_model + '_' + self.target + '.wav'
                    success_file_path = unique_wav_path(os.path.join('./SuccessDir', success_file_name))
                    sf.write(success_file_path, audio_numpy, 22050)

                converged, theta = self._estimate_threshold(conf_score, is_accepted)

                print(f"Threshold is now estimated at {theta}, the actual threshold is {threshold} \n")
                fitness_adv = -(max(theta, conf_score) - conf_score)

                fitness = fitness_adv + 0.02 * audio_quality

                print(f"[Individual {bird} Fitness: {fitness:.2f}]")
                print(f"[Individual {bird} conf_score: {conf_score:.2f}]")
                print(f"[Individual {bird} Adv: {fitness_adv:.2f}]")
                print(f"[Individual {bird} NISQA: {0.02 * audio_quality:.2f}]")
                print('\n')

            if count == 0 and epoch == 0:
                self.best_global = position
                self.best_global_fitness = fitness
            if fitness >= self.best_global_fitness:
                self.best_global = position
                self.best_global_fitness = fitness
            if fitness >= self.best_local[bird][1]:
                self.best_local[bird] = (position, fitness)



    def _update(self, position, bird, speed):
        w = 1
        c1 = 1 * random.random()
        c2 = 1 * random.random()
        speed = w * speed + c1 * (self.best_local[bird][0] - position) + c2 * (self.best_global - position)
        position += speed
        position = position.reshape(-1, 32)
        position = self._np_softmax(position) * 0.25
        position = position.flatten()
        return position, speed

    def run(self, iterations):
        self._initialize()
        for epoch in range(iterations):
            self._calculate_fitness(epoch)
            current_population = copy.deepcopy(self.population)
            for bird in current_population:
                position_o = current_population[bird][0]
                speed_o = current_population[bird][1]
                position_n, speed_n = self._update(position_o, bird, speed_o)
                self.population[bird] = (position_n, speed_n)

            print(f'epoch: {epoch+1} finish')

        if 'ASR' in self.target_model:
            return self.best_global

        elif 'SV' in self.target_model:
            return self.best_global, self.threshold_range




if __name__ == "__main__":
    reference_audio = './Original_TheyDenyTheyLied.wav'
    reference_text = "They deny they lied"
    # target_model can be 'googleASR' or 'iflytekASR' or 'gmmSV' or 'ivectorSV'
    target_model = 'gmmSV'
    # target can be speaker id or the target transcription
    target = "librispeech_p1089"
    # Run a small number of iterations with a small population size
    population_size = 20
    iterations = 10

    ga = PSO(reference_audio, reference_text, target_model, target, population_size)

    # Run the Genetic Algorithm
    ga.run(iterations)