from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def process_image(request):
    if request.method == "POST":
        print('Reueurururueueur')
        my_uploaded_file = request.FILES.get('my_uploaded_file')

        if not my_uploaded_file:
            return JsonResponse({'message': 'File Not Found'}, status=404)

        results = 'main(my_uploaded_file)'
    return JsonResponse(results, safe=True, status=True)

