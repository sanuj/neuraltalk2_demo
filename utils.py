import uuid, os
import subprocess

# Helper Functions
def handle_uploaded_image(f):
    filename = str(uuid.uuid4())
    current_dirname = os.path.dirname(os.path.abspath(__file__))
    filepath = current_dirname + '/images/' + filename + '/'
    os.mkdir(filepath)
    with open(filepath + f.name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

    # proc = subprocess.Popen('th ' + current_dirname + '/caption_image.lua -model ' + current_dirname + '/neuraltalk2/model_id1-501-1448236541.t7_cpu.t7 -image_folder ' +
    #     filepath + ' -num_images 1  -gpuid -1 -caption_file_name ' + filename, shell=True, stdout=subprocess.PIPE)
    # output = proc.communicate()[0]
    os.system('/home/sanuj/torch/install/bin/th ' + current_dirname + '/caption_image.lua -model ' + current_dirname + '/../model_id1-501-1448236541.t7_cpu.t7 -image_folder ' +
        filepath + ' -num_images 1  -gpuid -1 -caption_file_name ' + filename)

    caption_filepath = current_dirname + '/captions/' + filename + '.txt'
    caption_file = open(caption_filepath, 'r')
    output = caption_file.read()
    caption_file.close()

    os.remove(filepath + f.name)
    os.rmdir(filepath)
    os.remove(caption_filepath)

    return output
    # return 'th ' + current_dirname + '/caption_image.lua -model ' + current_dirname + '/neuraltalk2/model_id1-501-1448236541.t7_cpu.t7 -image_folder ' +
    #     filepath + ' -num_images 1  -gpuid -1 -caption_file_name ' + filename
