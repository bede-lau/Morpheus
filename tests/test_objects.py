"""Contract and native geometry tests. Model replies are fixtures, not live evidence."""
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from pipeline.object_server import match_objects, replacement_bundle, saved_scene, save_scene
from pipeline.simready import prepared_assets

class DecisionsContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        preview = self.cache/'prepared'/'plate_1'/'preview.jpg'
        preview.parent.mkdir(parents=True)
        preview.write_bytes(b'fixture thumbnail')
        self.assets = [{'id':'plate_1','category':'plate','name':'Fixture plate'}]
        self.images = ['data:image/jpeg;base64,YQ==']

    def test_classifies_then_compares_only_known_candidates(self):
        calls=[]
        def decide(body):
            calls.append(body)
            if len(calls)==1: return {'id':'test-category','answers':[{'name':'category','type':'choice','choice':'plate'}]}
            return {'id':'test-rank','answers':[
                {'name':'plate_1','type':'score','score':2.4,'confidence':.8},
                {'name':'invented_asset','type':'score','score':3},
                {'name':'plate_1','type':'score','score':float('nan')}]}
        with patch('pipeline.object_server.CACHE',self.cache):
            result=match_objects({'images':self.images},decide,self.assets)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0]['model'],'gpt-6-luna')
        self.assertEqual(calls[1]['questions'][0]['instructions'][1]['type'],'input_image')
        self.assertEqual([c['asset']['id'] for c in result['candidates']],['plate_1'])
        self.assertTrue(result['reviewRequired'])
        self.assertEqual(len(result['trace']),2)

    def test_refusal_does_not_choose_first_asset(self):
        result=match_objects({'images':self.images},lambda _: {'answers':[{'name':'category','type':'refusal'}]},self.assets)
        self.assertEqual(result['category'],'unknown')
        self.assertEqual(result['candidates'],[])

    def test_unknown_category_stops_retrieval(self):
        result=match_objects({'images':self.images},lambda _: {'answers':[{'name':'category','type':'choice','choice':'chair'}]},self.assets)
        self.assertEqual(result['candidates'],[])

    def test_rejects_remote_urls_and_unbounded_images(self):
        for images in [['http://localhost/private'], [], self.images*5, ['not-an-image']]:
            with self.subTest(images=images[:1]),self.assertRaises(ValueError):
                match_objects({'images':images},lambda _: self.fail('Unexpected model call'),self.assets)

class SavedSceneTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        cache=patch('pipeline.object_server.CACHE',Path(temporary.name))
        cache.start();self.addCleanup(cache.stop)
        assets=patch('pipeline.object_server.prepared_assets',return_value=[{'id':'plate_1','license':'CC-BY-4.0'}])
        assets.start();self.addCleanup(assets.stop)
        self.body={'version':1,'kind':'simready-replacements','units':'m','upAxis':'Z','baseRevision':0,
            'source':{'fileName':'room.spz','transform':{'matrix':[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],'matrixOrder':'column-major'}},
            'replacements':[{'id':'object-1','assetId':'plate_1','position':[1,2,3],'quaternion':[2,0,0,0],'scale':1.2,
                'selection':{'position':[1,2,3],'size':[.2,.2,.05]},'chosenBy':'manual'}]}

    def test_persists_canonical_scene_and_rejects_stale_writes(self):
        self.assertEqual(saved_scene()['revision'],0)
        result=save_scene(self.body)
        self.assertEqual(saved_scene(),result)
        self.assertEqual(result['revision'],1)
        self.assertEqual(result['replacements'][0]['quaternion'],[1,0,0,0])
        self.assertEqual(result['replacements'][0]['asset']['license'],'CC-BY-4.0')
        self.assertFalse(result['interactionValidated'])
        with self.assertRaisesRegex(ValueError,'another editor'):save_scene(self.body)
        self.assertEqual(saved_scene(),result)
        self.body.update(baseRevision=1,replacements=[])
        self.assertEqual(save_scene(self.body)['revision'],2)
        self.assertEqual(saved_scene()['replacements'],[])

    def test_rejects_invalid_masks_and_duplicate_ids_without_saving(self):
        import copy
        duplicate=copy.deepcopy(self.body)
        duplicate['replacements']*=2
        invalid=copy.deepcopy(self.body)
        invalid['replacements'][0]['selection']['size'][0]=0
        for body in [duplicate,invalid]:
            with self.assertRaises(ValueError):save_scene(body)
        self.assertEqual(saved_scene()['revision'],0)

class NativeAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets=prepared_assets()
        if not cls.assets: raise unittest.SkipTest('Prepare NVIDIA assets for the native integration tests.')

    def test_real_asset_export_preserves_pose_and_compiles(self):
        import mujoco
        for asset in self.assets:
            with self.subTest(asset=asset['id']):
                item={'assetId':asset['id'],'position':[.3,-.2,.8],
                      'quaternion':[math.cos(.3),0,0,math.sin(.3)],'scale':1.2}
                data=replacement_bundle({'replacements':[item]})
                archive=zipfile.ZipFile(io.BytesIO(data))
                report=json.loads(archive.read('validation.json'))
                self.assertEqual(report['compile'],'passed')
                self.assertEqual(report['interaction'],'not_validated')
                model=mujoco.MjModel.from_xml_string(archive.read('scene.xml').decode(),
                    {'assets/object_0.obj':archive.read('assets/object_0.obj')})
                import numpy as np
                np.testing.assert_allclose(model.body_pos[1],item['position'])
                np.testing.assert_allclose(model.body_quat[1],item['quaternion'])
                self.assertIn('assets/object_0.glb',archive.namelist())
                self.assertIn('credits/'+asset['id']+'/LICENSE.txt',archive.namelist())

    def test_rejects_invented_asset_and_bad_transforms(self):
        item={'assetId':self.assets[0]['id'],'position':[0,0,1],'quaternion':[1,0,0,0],'scale':1}
        for update in [{'assetId':'../../secret'},{'scale':0},{'quaternion':[0,0,0,0]},{'position':[float('nan'),0,0]}]:
            with self.subTest(update=update),self.assertRaises(ValueError):replacement_bundle({'replacements':[{**item,**update}]})

    def test_combines_existing_environment_without_duplicate_floor(self):
        environment={'version':1,'name':'test','units':'m','upAxis':'Z','objects':[
            {'id':'floor','type':'box','role':'floor','position':[0,0,-.05],'size':[4,4,.1]},
            {'id':'counter','type':'box','position':[1,0,.5],'size':[1,1,1]}]}
        item={'assetId':self.assets[0]['id'],'position':[1,0,1.1],'quaternion':[1,0,0,0],'scale':1}
        archive=zipfile.ZipFile(io.BytesIO(replacement_bundle({'replacements':[item],'environment':environment})))
        self.assertEqual(json.loads(archive.read('validation.json'))['geomCount'],3)

if __name__=='__main__': unittest.main()
