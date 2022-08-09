

from course.recommend import get_attractions, get_restaurants, get_hotels

from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializer import  CourseSerializer
# Create your views here.
@api_view(['GET'])
def get_course(request):
    
    # Query Params
    userId = int(request.GET['userId'])
    day = int(request.GET['day'])
    areaCode = int(request.GET['areaCode'])
    
    attractions = get_attractions(day, userId, areaCode)
    hotels = get_hotels(day, userId, areaCode)
    restaurants = get_restaurants(day, userId, areaCode)
    courses = []
    
    for i in range(day):
        course = [restaurants[0+(i*2)], attractions[0 + (3*i)],  attractions[1 + (3*i)],
                        restaurants[1+(i*2)], attractions[2 + (3*i)], hotels[0 + i]]
        
        if i == day - 1:
            course.pop()
        
        serializer = CourseSerializer(course, many = True) 
        courses.append(serializer.data)
        
        
    return Response(courses)

@api_view(['GET'])
def attractions(request):
    return get_attractions()

@api_view(['GET'])
def restaurants(request):
    return get_restaurants()


@api_view(['GET'])
def hotels(request):
    return get_hotels()
    
