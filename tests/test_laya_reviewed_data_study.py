"""Verify dataset adaptation, module isolation and unchanged CE configuration."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT
import laya_context_pair as original
from laya_reviewed_data_study import STUDY,data_rows,install


class ReviewedDataStudyTests(unittest.TestCase):
    def test_isolated_adapter_preserves_original_module_and_training_settings(self):
        before=original.STUDY
        revised=install(ROOT)
        self.assertEqual(original.STUDY,before)
        self.assertNotEqual(before,STUDY)
        self.assertEqual(revised.STUDY,STUDY)
        self.assertEqual(vars(revised.training_config(512)),vars(original.training_config(512)))
        with self.assertRaises(ValueError):revised.training_config(2048)

    def test_new_split_applies_only_user_changes_and_preserves_calibration(self):
        by,split=data_rows(ROOT)
        self.assertEqual(len(by),22497);self.assertEqual(len(split['gradient_train_ids']),22097)
        self.assertEqual(len(split['calibration_ids']),400)
        self.assertNotIn('train/neg/1892_3.txt',by)
        self.assertNotIn('train/neg/8443_4.txt',by)
        self.assertNotIn('train/pos/8512_7.txt',by)
        self.assertEqual(by['train/neg/1576_1.txt']['label'],0)  # clarified C012
        self.assertEqual(by['train/neg/2734_4.txt']['label'],1)
        self.assertEqual(by['train/pos/11585_10.txt']['label'],0)
        self.assertEqual(by['train/pos/773_7.txt']['label'],0)
        self.assertEqual(by['train/pos/90_7.txt']['label'],1)  # pending C086
        self.assertFalse(set(split['gradient_train_ids'])&set(split['calibration_ids']))


if __name__=='__main__':unittest.main()
