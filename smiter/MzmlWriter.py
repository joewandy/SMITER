"""
This file provides a class for writing mzML output from simulation.
For the actual generating of mzML file, the psims library is used.
"""
import os

import numpy as np
from loguru import logger
from psims.mzml.writer import MzMLWriter as PsimsMzMLWriter
import pathlib

INITIAL_SCAN_ID = 100000
DEFAULT_MS1_SCAN_WINDOW = (70.0, 1000.0)
POSITIVE = 'Positive'
NEGATIVE = 'Negative'


def create_if_not_exist(out_dir):
    """
    Creates a directory if it doesn't already exist
    Args:
        out_dir: the directory to create, if it doesn't exist

    Returns: None.

    """
    if not pathlib.Path(out_dir).exists():
        logger.info('Created %s' % out_dir)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)


def compute_isolation_windows(isolation_width_list, precursor_mz_list):
    isolation_windows = []
    windows = []
    for i, mz in enumerate(precursor_mz_list):
        isolation_width = isolation_width_list[i]
        mz_lower = mz - (isolation_width / 2)
        mz_upper = mz + (isolation_width / 2)
        windows.append((mz_lower, mz_upper))
    isolation_windows.append(windows)
    return isolation_windows


class VScan():
    """
    A class to store scan information
    """

    def __init__(self, scan_id, mzs, intensities, ms_level, rt,
                 scan_duration=None, scan_params=None, parent=None,
                 fragevent=None):
        """
        Creates a scan

        Args:
            scan_id: current scan id
            mzs: an array of mz values
            intensities: an array of intensity values
            ms_level: the ms level of this scan
            rt: the retention time of this scan
            scan_duration: how long this scan takes, if known.
            scan_params: the parameters used to generate this scan, if known
            parent: parent precursor peak, if known
            fragevent: fragmentation event associated to this scan, if any
        """
        assert len(mzs) == len(intensities)
        self.scan_id = scan_id

        # ensure that mzs and intensites are sorted by their mz values
        p = mzs.argsort()
        self.mzs = mzs[p]
        self.intensities = intensities[p]

        self.ms_level = ms_level
        self.rt = rt
        self.num_peaks = len(mzs)

        self.scan_duration = scan_duration
        self.scan_params = scan_params
        self.parent = parent
        self.fragevent = fragevent

    @classmethod
    def from_mzmlscan(self, scan):
        mzs, intensities = zip(*scan.peaks)
        return VScan(
            scan_id=scan.scan_no,
            mzs=np.array(mzs),
            intensities=np.array(intensities),
            ms_level=scan.ms_level,
            rt=scan.rt_in_seconds
        )

    def __repr__(self):
        return 'Scan %d num_peaks=%d rt=%.2f ms_level=%d' % (
            self.scan_id, self.num_peaks, self.rt, self.ms_level)


class ScanParameters():
    """
    A class to store parameters used to instruct the mass spec how to
    generate a scan. This object is usually created by the controller.
    It is used by the controller to instruct the mass spec what actions (scans)
    to perform next.
    """

    MS_LEVEL = 'ms_level'
    COLLISION_ENERGY = 'collision_energy'
    POLARITY = 'polarity'
    FIRST_MASS = 'first_mass'
    LAST_MASS = 'last_mass'
    ORBITRAP_RESOLUTION = 'orbitrap_resolution'
    AGC_TARGET = 'agc_target'
    MAX_IT = 'max_it'
    MASS_ANALYSER = 'analyzer'
    ACTIVATION_TYPE = 'activation_type'
    ISOLATION_MODE = 'isolation_mode'
    SOURCE_CID_ENERGY = 'source_cid_energy'
    METADATA = 'metadata'
    UNIQUENESS_TOKEN = "uniqueness_token"

    # this is used for DIA-based controllers to specify which windows to
    # fragment
    ISOLATION_WINDOWS = 'isolation_windows'

    # precursor m/z and isolation width have to be specified together for
    # DDA-based controllers
    PRECURSOR_MZ = 'precursor_mz'
    ISOLATION_WIDTH = 'isolation_width'

    # used in Top-N, hybrid and ROI controllers to perform dynamic exclusion
    DYNAMIC_EXCLUSION_MZ_TOL = 'dew_mz_tol'
    DYNAMIC_EXCLUSION_RT_TOL = 'dew_rt_tol'

    # only used by the hybrid controller for now, since its Top-N may change
    # depending on time for other DDA controllers it's always the same
    # throughout the whole run, so we don't send this parameter
    CURRENT_TOP_N = 'current_top_N'

    # if the scan id is specified, then it should be used by the mass spec
    # useful for pre-scheduled controllers where we want the controller
    # to know the scan ids of MS1, MS2
    # and also the precursor ids of those MS2 scans in advance.
    SCAN_ID = 'scan_id'

    def __init__(self):
        """
        Create a scan parameter object
        """
        self.params = {}

    def set(self, key, value):
        """
        Set scan parameter value

        Args:
            key: a scan parameter name
            value: a scan parameter value

        Returns: None
        """
        self.params[key] = value

    def get(self, key):
        """
        Gets scan parameter value

        Args:
            key: the key to look for

        Returns: the corresponding value in this ScanParameter
        """
        if key in self.params:
            return self.params[key]
        else:
            return None

    def get_all(self):
        """
        Get all scan parameters
        Returns: all the scan parameters
        """
        return self.params

    def compute_isolation_windows(self):
        """
        Gets the full-width (DDA) isolation window around a precursor m/z
        """
        precursor_list = self.get(ScanParameters.PRECURSOR_MZ)
        precursor_mz_list = [precursor.precursor_mz for precursor in precursor_list]
        isolation_width_list = self.get(ScanParameters.ISOLATION_WIDTH)
        return compute_isolation_windows(isolation_width_list, precursor_mz_list)

    def __repr__(self):
        return 'ScanParameters %s' % (self.params)


class Precursor():
    """
    A class to store precursor peak information when writing an MS2 scan.
    """

    def __init__(self, precursor_mz, precursor_intensity, precursor_charge,
                 precursor_scan_id):
        """
        Create a Precursor object.
        Args:
            precursor_mz: the m/z value of this precursor peak.
            precursor_intensity: the intensity value of this precursor peak.
            precursor_charge: the charge of this precursor peak
            precursor_scan_id: the assocated MS1 scan ID that contains this precursor peak
        """
        self.precursor_mz = precursor_mz
        self.precursor_intensity = precursor_intensity
        self.precursor_charge = precursor_charge
        self.precursor_scan_id = precursor_scan_id

    def __repr__(self):
        return 'Precursor mz %f intensity %f charge %d scan_id %d' % (
            self.precursor_mz, self.precursor_intensity, self.precursor_charge,
            self.precursor_scan_id)

class MzmlWriter():
    """
    A class to write peak data to mzML file, typically called after running simulation.
    """

    def __init__(self, analysis_name, scans):
        """
        Initialises the mzML writer class.

        Args:
            analysis_name: Name of the analysis.
            scans: dict where key is scan level, value is a list of Scans object for that level.
        """
        self.analysis_name = analysis_name
        self.scans = scans

    def write_mzML(self, out_file):
        """
        Write mzMl to output file

        Args:
            out_file: path to mzML file

        Returns: None

        """
        # if directory doesn't exist, create it
        out_dir = os.path.dirname(out_file)
        create_if_not_exist(out_dir)

        # start writing mzML here
        with PsimsMzMLWriter(open(out_file, 'wb')) as writer:
            # add default controlled vocabularies
            writer.controlled_vocabularies()

            # write other fields like sample list, software list, etc.
            self._write_info(writer)

            # open the run
            with writer.run(id=self.analysis_name):
                self._write_spectra(writer, self.scans)

                # open chromatogram list sections
                with writer.chromatogram_list(count=1):
                    tic_rts, tic_intensities = self._get_tic_chromatogram(
                        self.scans)
                    writer.write_chromatogram(
                        tic_rts, tic_intensities, id='tic',
                        chromatogram_type='total ion current chromatogram',
                        time_unit='second')

        writer.close()

    def _write_info(self, out):
        """
        Write various information to output stream
        Args:
            out: the output stream from psims

        Returns: None

        """
        # check file contains what kind of spectra
        has_ms1_spectrum = 1 in self.scans
        has_msn_spectrum = 1 in self.scans and len(self.scans) > 1
        file_contents = [
            'centroid spectrum'
        ]
        if has_ms1_spectrum:
            file_contents.append('MS1 spectrum')
        if has_msn_spectrum:
            file_contents.append('MSn spectrum')
        out.file_description(
            file_contents=file_contents,
            source_files=[]
        )
        out.sample_list(samples=[])
        out.software_list(software_list={
            'id': 'VMS',
            'version': '1.0.0'
        })
        out.scan_settings_list(scan_settings=[])
        out.instrument_configuration_list(instrument_configurations={
            'id': 'VMS',
            'component_list': []
        })
        out.data_processing_list({'id': 'VMS'})

    def sort_filter(self, all_scans, min_scan_id):
        """
        Filter scans according to certain criteria. Currently it filters by
        the minimum scan ID, as a workaround to IAPI which produces unwanted scans at
        low scan IDs.

        Args:
            all_scans: the list of scans to filter
            min_scan_id: the minimum scan ID to filter

        Returns: the list of filtered scans

        """
        all_scans = sorted(all_scans, key=lambda x: x.rt)
        all_scans = [x for x in all_scans if x.num_peaks > 0]
        all_scans = list(filter(lambda x: x.scan_id >= min_scan_id, all_scans))

        # FIXME: why do we need to do this???!!
        # add a single peak to empty scans
        # empty = [x for x in all_scans if x.num_peaks == 0]
        # for scan in empty:
        #     scan.mzs = np.array([100.0])
        #     scan.intensities = np.array([1.0])
        #     scan.num_peaks = 1
        return all_scans

    def _write_spectra(self, writer, scans, min_scan_id=INITIAL_SCAN_ID):
        """
        Helper method to actually write a collection of spectra
        Args:
            writer: the output stream from psims
            scans: a list of scans
            min_scan_id: the minimum scan ID to write

        Returns: None

        """
        # NOTE: we only support writing up to ms2 scans for now
        assert len(scans) <= 3

        # get all scans across different ms_levels and sort them by scan_id
        all_scans = []
        for ms_level in scans:
            all_scans.extend(scans[ms_level])
        all_scans = self.sort_filter(all_scans, min_scan_id)
        spectrum_count = len(all_scans)

        # write scans
        with writer.spectrum_list(count=spectrum_count):
            for scan in all_scans:
                self._write_scan(writer, scan)

    def _write_scan(self, out, scan):
        """
        Helper method to write a single scan
        Args:
            out: the psims output stream
            scan: the scan to write

        Returns: None

        """
        assert scan.num_peaks > 0
        label = 'MS1 Spectrum' if scan.ms_level == 1 else 'MSn Spectrum'
        precursor_information = None
        if scan.ms_level == 2:
            collision_energy = scan.scan_params.get(
                ScanParameters.COLLISION_ENERGY)
            activation_type = scan.scan_params.get(
                ScanParameters.ACTIVATION_TYPE)

            precursor_information = []
            for precursor in scan.scan_params.get(ScanParameters.PRECURSOR_MZ):
                precursor_information.append({
                    "mz": precursor.precursor_mz,
                    "intensity": precursor.precursor_intensity,
                    "charge": precursor.precursor_charge,
                    "spectrum_reference": precursor.precursor_scan_id,
                    "activation": [activation_type,
                                   {"collision energy": collision_energy}]
                })

        lowest_observed_mz = min(scan.mzs)
        highest_observed_mz = max(scan.mzs)
        # bp_pos = np.argmax(scan.intensities)
        # bp_intensity = scan.intensities[bp_pos]
        # bp_mz = scan.mzs[bp_pos]
        scan_id = scan.scan_id

        try:
            first_mz = scan.scan_params.get(ScanParameters.FIRST_MASS)
            last_mz = scan.scan_params.get(ScanParameters.LAST_MASS)
        # if it's a method scan (not a custom scan), there's no scan_params
        # to get first_mz and last_mz
        except AttributeError:
            first_mz, last_mz = DEFAULT_MS1_SCAN_WINDOW

        polarity = scan.scan_params.get(ScanParameters.POLARITY)
        if polarity == POSITIVE:
            int_polarity = 1
        elif polarity == NEGATIVE:
            int_polarity = -1
        else:
            int_polarity = 1
            logger.warning(
                "Unknown polarity in mzml writer: {}".format(polarity))

        out.write_spectrum(
            scan.mzs, scan.intensities,
            id=scan_id,
            polarity=int_polarity,
            centroided=True,
            scan_start_time=scan.rt / 60.0,
            scan_window_list=[(first_mz, last_mz)],
            params=[
                {label: ''},
                {'ms level': scan.ms_level},
                {'total ion current': np.sum(scan.intensities)},
                {'lowest observed m/z': lowest_observed_mz},
                {'highest observed m/z': highest_observed_mz},
                # {'base peak m/z', bp_mz},
                # {'base peak intensity', bp_intensity}
            ],
            precursor_information=precursor_information
        )

    def _get_tic_chromatogram(self, scans):
        """
        Helper method to write total ion chromatogram information
        Args:
            scans: the list of scans

        Returns: a tuple of time array and intensity arrays for the TIC

        """
        time_array = []
        intensity_array = []
        for ms1_scan in scans[1]:
            time_array.append(ms1_scan.rt)
            intensity_array.append(np.sum(ms1_scan.intensities))
        time_array = np.array(time_array)
        intensity_array = np.array(intensity_array)
        return time_array, intensity_array
