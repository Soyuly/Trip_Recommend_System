from dataclasses import field
from rest_framework import serializers

class Course(object):
    def __init__(self,code, areaCode, id, name, rating, lat, lng):
        self.code = code
        self.areaCode = areaCode
        self.id= id
        self.name = name
        self.rating = rating
        self.lat = lat
        self.lng = lng
        

class CourseSerializer(serializers.Serializer):
    code = serializers.ChoiceField(choices=["관광지", "식당", "호텔"])
    areaCode = serializers.IntegerField()
    id = serializers.IntegerField()
    name = serializers.CharField(max_length = 500)
    rating = serializers.FloatField()
    lat = serializers.DecimalField(10,8)
    lng = serializers.DecimalField(10,7)
    class Meta:
        model = Course
        fields = '__all__' 