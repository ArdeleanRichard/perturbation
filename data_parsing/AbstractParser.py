import numpy as np


class AbstractParser:
    def __init__(self):
        self.TIMESTAMP_LENGTH = 1

        self.groups = None
        self.trial_timestamp_intervals = None

    def get_index_line(self, lines, const_string):
        index_line = np.where(lines == const_string)
        return index_line[0][0] + 1

    def get_index_line2(self, lines, const_string1, const_string2):
        if np.count_nonzero(lines == const_string1) == 0:
            index_line = np.where(lines == const_string2)
        else:
            index_line = np.where(lines == const_string1)
        return index_line[0][0] + 1

    def split_event_codes(self, event_codes, TRIAL_START, STIMULUS_ON, STIMULUS_OFF, TRIAL_END):
        groups = []

        group = []
        # print(np.unique(self.event_codes))
        for id, event_code in enumerate(event_codes):
            if event_code == TRIAL_START:
                group = []
                group.append(id)
            elif len(group) == 1 and event_code == STIMULUS_ON:
                group.append(id)
            elif len(group) == 2 and event_code == STIMULUS_OFF:
                group.append(id)
            elif len(group) == 3 and event_code == TRIAL_END:
                group.append(id)
                groups.append(group)
                group = []

        self.groups = np.array(groups)

    def split_event_timestamps_by_codes(self, event_timestamps):
        timestamp_intervals = []
        for group in self.groups:
            # STIMULUS EVENT CODES TIMESTAMPS LENGTH
            print(self.event_timestamps[group[0]], self.event_timestamps[group[1]] - self.event_timestamps[group[0]], self.event_timestamps[group[3]] - self.event_timestamps[group[2]],self.event_timestamps[group[3]])
            timestamps_of_interest = [event_timestamps[group[0]], event_timestamps[group[-1]]]
            timestamp_intervals.append(timestamps_of_interest)

        timestamp_trial_intervals = np.array(timestamp_intervals)

        # check timestamp intervals, ensure same length as minimum trial, stimulus on considered ok
        flag = False
        len_check = timestamp_trial_intervals[0, 1] - timestamp_trial_intervals[0, 0]
        lens = []
        for timestamp_interval in timestamp_trial_intervals:
            if timestamp_interval[1] - timestamp_interval[0] != len_check:
                flag = True
            lens.append(timestamp_interval[1] - timestamp_interval[0])

        min_len = min(lens)

        if flag == True:
            for timestamp_interval in timestamp_trial_intervals:
                if timestamp_interval[1] - timestamp_interval[0] != min_len:
                    timestamp_interval[1] = timestamp_interval[0] + min_len

        # print(self.timestamp_trial_intervals[:, 1] - self.timestamp_trial_intervals[:, 0])
        self.trial_timestamp_intervals = timestamp_trial_intervals
        self.NR_TRIALS = len(timestamp_trial_intervals)

    def get_intracellular_labels(self):
        intracellular_labels = np.zeros((len(self.timestamps)))
        # given_index = np.zeros((len(event_timestamps[event_codes == 1])))
        # for index, timestamp in enumerate(timestamps):
        #     for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
        #         if event_timestamp - 1000 < timestamp < event_timestamp + 1000 and given_index[index2] == 0:
        #             given_index[index2] = 1
        #             intracellular_labels[index] = 1
        #             break

        for index2, event_timestamp in enumerate(self.event_timestamps[self.event_codes == 1]):
            indexes = []
            for index, timestamp in enumerate(self.timestamps):
                if event_timestamp - self.WAVEFORM_LENGTH < timestamp < event_timestamp + self.WAVEFORM_LENGTH:
                    # given_index[index2] = 1
                    indexes.append(index)

            if indexes != []:
                min = indexes[0]
                for i in range(1, len(indexes)):
                    if self.timestamps[indexes[i]] < self.timestamps[min]:
                        min = indexes[i]
                intracellular_labels[min] = 1

        return intracellular_labels.astype(int)