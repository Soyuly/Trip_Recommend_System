from dataclasses import field
from rest_framework import serializers

class Course(object):
    def __init__(self, id, name, rating, lat, lng):
        self.id= id
        self.name = name
        self.rating = rating
        self.lat = lat
        self.lng = lng
        

class CourseSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField(max_length = 500)
    rating = serializers.FloatField()
    lat = serializers.DecimalField(10,8)
    lng = serializers.DecimalField(10,7)
    class Meta:
        model = Course
        fields = '__all__' 