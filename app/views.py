from django.shortcuts import render
from django.http import HttpResponse

from .forms import ImageUploadForm
from utils import handle_uploaded_image

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            caption = handle_uploaded_image(request.FILES['image'])
            # return HttpResponseRedirect('/success/url/')
            return HttpResponse(caption)
    else:
        form = ImageUploadForm()

    return render(request, 'app/image_upload.html', { 'form': form })
