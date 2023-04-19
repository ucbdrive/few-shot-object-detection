
import sys
import os
import flask
import zipfile
import yaml

from PIL import Image
from flask_cors import CORS

from types import SimpleNamespace
from fsdettools.train_fs import train_fs

from multiprocessing import Process


processlist = []
model_store = './docker/model_store'


def package_model(args, manifest):
    """
    Internal helper for the exporting model command line interface.
    """
    model_file = args.model_file
    serialized_file = args.serialized_file
    model_name = args.model_name
    handler = args.handler
    extra_files = args.extra_files
    export_file_path = args.export_path
    requirements_file = args.requirements_file

    try:
        ModelExportUtils.validate_inputs(model_name, export_file_path)
        # Step 1 : Check if .mar already exists with the given model name
        export_file_path = ModelExportUtils.check_mar_already_exists(
            model_name, export_file_path, args.force, args.archive_format
        )

        # Step 2 : Copy all artifacts to temp directory
        artifact_files = {
            "model_file": model_file,
            "serialized_file": serialized_file,
            "handler": handler,
            "extra_files": extra_files,
            "requirements-file": requirements_file,
        }

        model_path = ModelExportUtils.copy_artifacts(model_name, **artifact_files)

        # Step 2 : Zip 'em all up
        ModelExportUtils.archive(
            export_file_path, model_name, model_path, manifest, args.archive_format
        )
        shutil.rmtree(model_path)
        logging.info(
            "Successfully exported model %s to file %s", model_name, export_file_path
        )
    except ModelArchiverError as e:
        logging.error(e)
        sys.exit(1)


def generate_model_archive():
    """
    Generate a model archive file
    :return:
    """

    logging.basicConfig(format="%(levelname)s - %(message)s")
    args = ArgParser.export_model_args_parser().parse_args()
    manifest = ModelExportUtils.generate_manifest_json(args)
    package_model(args, manifest=manifest)


def get_log_path(name):
    return os.path.join(os.getenv("FSDET_ROOT"),'logs',name+'.log')

def training_worker(name,jobfilepath,basemodelfile):

    global model_store

    print("logging to "+get_log_path(name))

    sys.stdout = open(get_log_path(name), "w")

    args = SimpleNamespace(datasetconfig=jobfilepath , ignoreunknown=True, splitfact=-1)

    os.chdir(os.getenv("FSDET_ROOT"))
    
    if os.getenv("MODEL_STORE"):
        model_store = os.getenv("MODEL_STORE")

    # run the training pipeline
    train_fs(args)
    
    # build the archive
    basename = basemodelfile.split('/')
    newmodelname = ''
    for i in range(len(basename)-2):
        if basename[i]=='coco':
            basename[i] = 'fs'
        newmodelname = newmodelname + basename[i]
        newmodelname = newmodelname + '/'
        
    modeldir = '_'.join(basename[-2].split('_')[:-1]) + '_' + name
    
    newmodelname = newmodelname + modeldir + '/' + basename[-1]
        
    # TODO add dependent configs
    os.system("torch-model-archiver -f --model-name "+name+" --handler ./docker/fsod_handler.py --extra-files "+jobfilepath+ " --export-path "+model_store+" -v 0.1 --serialized-file "+newmodelname)
    
    # register with torchserve
    # - unregister in case it already exists
    
    os.system("curl -X DELETE http://localhost:8081/models/"+name)
    
    os.system("curl -X POST  \"http://localhost:8081/models?url="+model_store+"/"+name+".mar&name="+name+"\"")

    

def main():
    app = flask.Flask(__name__)
    CORS(app)
    
    def store_file_to_path(relfilename,data,binary=False):
        
        filename = os.path.join(os.getenv("FSDET_ROOT"),relfilename)
        
        relpath = os.path.dirname(filename)
        
        os.makedirs(relpath,exist_ok=True)
        
        print("storing "+filename)
        
        mode = 'w'
        if binary:
            mode = 'wb'

        with open(filename,mode) as f:
            f.write(data)    
        
    @app.route('/store', methods=['POST'])
    def store_file():
        headers = flask.request.headers

        print( "Request headers:\n" + str(headers) )
        """Print posted body to stdout"""

        try: 
            data = flask.request.data.decode('utf-8')

            filename = flask.request.args.get('name')
        
            store_file_to_path(filename,data)

        except Exception as e:
            return flask.Response(response='Failed to store file: '+str(e), status=500)

            
        return flask.Response(status=200)
        
    @app.route('/train', methods=['POST'])
    def train():

        global processlist

        headers = flask.request.headers
       

        print( "Request headers:\n" + str(headers) )
        """Print posted body to stdout"""

        cfg_file_names = []
        cfg_files = []
        img_file_names = []
        img_files = []

        try: 
            cfgzip = flask.request.files['config']              
            file_like_object = cfgzip.stream._file        
            zipfile_ob = zipfile.ZipFile(file_like_object)
            cfg_file_names = zipfile_ob.namelist()
    
            cfg_files = [(zipfile_ob.open(name).read(),name) for name in cfg_file_names]
            
            # check if images have also been provided
            if 'images' in flask.request.files:
                imgzip = flask.request.files['images']              
                file_like_object = imgzip.stream._file        
                zipfile_ob = zipfile.ZipFile(file_like_object)
                img_file_names = zipfile_ob.namelist()
    
                img_files = [(zipfile_ob.open(name).read(),name) for name in img_file_names]                
            
        except Exception as e:
            return flask.Response(response='Failed to parse ZIP file: '+str(e), status=500)

            
        # process config files
        jsoncontent = '{}'
        jobcfg = {}
        jobfilepath = ''
        basemodelfile = ''
        
        try:
            for cf,cfdata in zip(cfg_file_names,cfg_files):
                if cf.endswith(".yaml"):
                    bn = os.path.basename(cf)
                    jobfilepath = os.path.join('configs','custom_datasets',bn)
                    
                    store_file_to_path(jobfilepath,cfdata[0].decode('utf-8'))
                   
                    jobcfg = yaml.safe_load(cfdata[0].decode('utf-8'))
                elif cf.endswith(".json"):
                    
                    jsoncontent = cfdata
                else:
                    print('file '+cf+' ignored')
         
            # now serialise JSON, as name can be taken from YAML
            target_fn = jobcfg['novel']['data']
            store_file_to_path(target_fn,jsoncontent[0].decode('utf-8'))
            
            basemodelfile = jobcfg['base']['model']
            
        except Exception as e:
            return flask.Response(response='Failed to process config files: '+str(e), status=500)
            
        # store images
        
        try:
            dataroot = jobcfg['novel']['data_dir']
        
            for imf,imfdata in zip(img_file_names,img_files):
                target_fn = os.path.join('datasets',dataroot,imf)
                store_file_to_path(target_fn,imfdata[0],True)
    
            
        except Exception as e:
            return flask.Response(response='Failed to process image files: '+str(e), status=500)            
            
        # clean up processes
        newprocesslist = []
        for p in processlist:
            if p.is_alive():
                newprocesslist.append(p)
                
        processlist = newprocesslist
            
        # start training in own thread

        try: 
            p = Process(target=training_worker, args=(jobcfg['name'],jobfilepath,basemodelfile,))
            processlist.append(p)
            p.start()
        
        except Exception as e:
            return flask.Response(response='Failed to start worker process: '+str(e), status=500)            
           
            
        return flask.Response(status=200)           
    
        
    @app.route('/log', methods=['GET'])
    def get_log():

        args = flask.request.args

        name = args.get('name',type=str,default='')
        lastlines = args.get('tail',type=int,default=0)
        
        logfilename = get_log_path(name)
                
        try:
            lines = []
            with open(logfilename) as f:
                lines = f.readlines()
                                
            flines = lines[-lastlines:]
                        
            logcontent = '\n'.join(flines)
           
        
        except Exception as e:
            return flask.Response(response='Log file not found: '+str(name),status=404)            
        

        return flask.Response(response=logcontent,status=200,mimetype='text/plain')

    
    @app.errorhandler(404)
    def handle_404(e):
        # handle all other routes here
        return flask.Response(status=404)          

    retval = app.run(debug=True, host='::', port=3010)
    
    return retval


if __name__ == '__main__':
    sys.exit(main())
