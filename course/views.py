

from course.recommend import get_attractions, get_restaurants, get_hotels

from rest_framework.decorators import api_view


# Create your views here.
@api_view(['GET'])
def attractions(request):
    return get_attractions()

@api_view(['GET'])
def restaurants(request):
    return get_restaurants()


@api_view(['GET'])
def hotels(request):
    return get_hotels()
    
