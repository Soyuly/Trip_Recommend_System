from surprise import Reader, SVD
import pandas as pd
from surprise.dataset import DatasetAutoFolds
from .serializer import Course, CourseSerializer
import os

module_dir = os.path.dirname(__file__)  
attraction_csv = os.path.join(module_dir, 'csv/관광지.csv')
attractionRatings_csv = os.path.join(module_dir, 'csv/관광지평점.csv')
restaurant_csv = os.path.join(module_dir, 'csv/맛집.csv')
restaurantRatings_csv = os.path.join(module_dir, 'csv/맛집평점.csv')
hotel_csv = os.path.join(module_dir, 'csv/숙박.csv')
hotelRatings_csv = os.path.join(module_dir, 'csv/숙박평점.csv')

def get_attractions(day, userId, areaCode):
    
    # 관광지 평점을 불러서, 헤더 없는 데이터프레임으로 저장을 한다.
    df = pd.read_csv(attractionRatings_csv)
    df.to_csv(path_or_buf = "./course/csv/관광지평점_헤더X.csv", index=False, header=False)
    
    # 다시 숙박평점 헤더가 없는 평점 데이터프레임을 불러온다.
    attractionRatings_noh_csv = os.path.join(module_dir, 'csv/관광지평점_헤더X.csv')
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0.5,5))
    
    # 전부 학습 데이터를 사용해서, AI에게 학습시킨다.
    data_folds = DatasetAutoFolds(ratings_file=attractionRatings_noh_csv, reader=reader)
    trainset = data_folds.build_full_trainset()
    ai = SVD(n_epochs=20, n_factors = 50, random_state = 0)
    ai.fit(trainset)

    # 평점과 관광지 데이터프레임을 읽는다.
    ratings = pd.read_csv(attractionRatings_csv)
    attractions = pd.read_csv(attraction_csv)
    
    # userId에 대해 방문 하지 않은 관광지들을 구한다.
    unVisitedAttractions = getUnvisitedAttractions(ratings, attractions, userId)
    print(areaCode)
    # 지역별로 추천 받는다.
    topAttractionPreds = recommendAttractions(ai, userId, unVisitedAttractions, attractions, areaCode, top=50 * day)
    
    # 만약에 지역별 데이터가 3개 미만이면, 다시 추천을 받는다.
    searchScale = 2
    while len(topAttractionPreds) < 3:
        topAttractionPreds = recommendAttractions(ai, userId, unVisitedAttractions, attractions, areaCode, top=50 * day * searchScale )
        searchScale += 1
        print(topAttractionPreds)
        
    
    return topAttractionPreds[:5 * day]

def get_restaurants(day, userId, areaCode):
    # 관광지 평점을 불러서, 헤더 없는 데이터프레임으로 저장을 한다.
    df = pd.read_csv(restaurantRatings_csv)
    df.to_csv(path_or_buf = "./course/csv/맛집평점_헤더X.csv", index=False, header=False)
    
    # 다시 숙박평점 헤더가 없는 평점 데이터프레임을 불러온다.
    restaurantRatings_noh_csv = os.path.join(module_dir, 'csv/맛집평점_헤더X.csv')
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0.5,5))
    
    # 전부 학습 데이터를 사용해서, AI에게 학습시킨다.
    data_folds = DatasetAutoFolds(ratings_file=restaurantRatings_noh_csv, reader=reader)
    trainset = data_folds.build_full_trainset()
    ai = SVD(n_epochs=20, n_factors = 50, random_state = 0)
    ai.fit(trainset)

    # 평점과 관광지 데이터프레임을 읽는다.
    ratings = pd.read_csv(restaurantRatings_csv)
    restaurants = pd.read_csv(restaurant_csv)
    
    # userId에 대해 방문 하지 않은 관광지들을 구한다.
    unVisitedRestaurants = getUnvisitedRestaurants(ratings, restaurants, userId)
    
    # 지역별로 추천 받는다.
    topRestaurantPreds = recommendRestaurants(ai, userId, unVisitedRestaurants, restaurants, areaCode, top=50*day)

    # 만약에 지역별 데이터가 3개 미만이면, 다시 추천을 받는다.
    while len(topRestaurantPreds) < 3:
        topRestaurantPreds = recommendRestaurants(ai, userId, unVisitedRestaurants, restaurants, areaCode, top=70*day)
        
    return topRestaurantPreds[:3*day]

def get_hotels(day, userId, areaCode):
    
    # 관광지 평점을 불러서, 헤더 없는 데이터프레임으로 저장을 한다.
    df = pd.read_csv(hotelRatings_csv)
    df.to_csv(path_or_buf = "./course/csv/숙박평점_헤더X.csv", index=False, header=False)
    
    # 다시 숙박평점 헤더가 없는 평점 데이터프레임을 불러온다.
    hotelRatings_noh_csv = os.path.join(module_dir, 'csv/숙박평점_헤더X.csv')
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0.5,5))
    
    # 전부 학습 데이터를 사용해서, AI에게 학습시킨다.
    data_folds = DatasetAutoFolds(ratings_file=hotelRatings_noh_csv, reader=reader)
    trainset = data_folds.build_full_trainset()
    ai = SVD(n_epochs=20, n_factors = 50, random_state = 0)
    ai.fit(trainset)

    # 평점과 관광지 데이터프레임을 읽는다.
    ratings = pd.read_csv(hotelRatings_csv)
    hotels = pd.read_csv(hotel_csv)
    
    # userId에 대해 방문 하지 않은 관광지들을 구한다.
    unVisitedHotels = getUnvisitedHotels(ratings, hotels, userId)
    
    # 지역별로 추천 받는다.
    topHotelPreds = recommendHotels(ai, userId, unVisitedHotels, hotels, areaCode, top=50*day)
    
     # 만약에 지역별 데이터가 3개 미만이면, 다시 추천을 받는다.
    while len(topHotelPreds) < 3:
        topHotelPreds = recommendHotels(ai, 5, unVisitedHotels, hotels, 2, top=70*day)
        
    
    return topHotelPreds[:3*day]

# 아직 안 가본 명소 찾기
def getUnvisitedAttractions(ratings, attractions, userId):
    
    # 방문한 명소들
    visitedAttractions = ratings[ratings['userId'] == userId]['itemId'].tolist()
    
    # 전체 명소들
    totalAttractions = attractions['ID'].tolist()
    
    # 방문하지 않은 명소들
    unVisitedAttractions = [attraction for attraction in totalAttractions if attraction not in visitedAttractions]
    print(f'평점 매긴 관광지 수 : {len(visitedAttractions)}, 추천대상 관광지 수 : {len(unVisitedAttractions)}, 전체 관광지 수 :  {len(totalAttractions)}')
    
    return unVisitedAttractions

# 아직 안 가본 식당 찾기
def getUnvisitedRestaurants(ratings, restaurants, userId):
    
    # 방문한 식당들
    visitedRestaurants = ratings[ratings['userId'] == userId]['itemId'].tolist()
    
    # 전체 식당들
    totalRestaurants = restaurants['ID'].tolist()
    
    # 방문하지 않은 식당들
    unVisitedRestaurants = [attraction for attraction in totalRestaurants if attraction not in visitedRestaurants]
    print(f'평점 매긴 관광지 수 : {len(visitedRestaurants)}, 추천대상 관광지 수 : {len(unVisitedRestaurants)}, 전체 관광지 수 :  {len(totalRestaurants)}')
    
    return unVisitedRestaurants

# 아직 안 가본 호텔 찾기
def getUnvisitedHotels(ratings, hotels, userId):
    
    
    visiteHotels = ratings[ratings['userId'] == userId]['itemId'].tolist()
    
    totalHotels = hotels['ID'].tolist()
    
    unVisiteHotels = [attraction for attraction in totalHotels if attraction not in visiteHotels]
    print(f'평점 매긴 관광지 수 : {len(visiteHotels)}, 추천대상 관광지 수 : {len(unVisiteHotels)}, 전체 관광지 수 :  {len(totalHotels)}')
    
    return unVisiteHotels

def recommendAttractions(ai, userId, unVisitedAttractions, attractions, areaId, top):
    # AI 객체의 predict()를 이용해서 특정 userId의 평점이 없는 관광지에 대해 평점 예측
    predictions = [ai.predict(str(userId), str(attractionId)) for attractionId in unVisitedAttractions]
    
    # predictions는 Predction() 하나의 객체르 되어 있기 때문에 예측 평점(est값)을 기준으로 정렬 헤야한다.
    # est값을 반환하는 함수 정의
    def sortedkey_est(pred):
        return pred.est
    
    # sortkey_est 함수로 리스트를 정렬하는 sort 함수의 key인자에 넣어주자
    # 리스트 sort는 디폴트 값이 inplace=True인 것처럼 정렬되어 나온다. reverse=True가 내림차순
    predictions.sort(key=sortedkey_est, reverse=True)
    
    # 상위 n개의 예측값들만 할당
    topPredictions = predictions[:top]
    
    # topPredictions에서 itemId, rating, 명소 이름 뽑아내기
    topAttractionIds = [int(pred.iid) for pred in topPredictions]
    topAttractionRatings = [pred.est for pred in topPredictions]
    topAttraction = attractions[(attractions.ID.isin(topAttractionIds)) & (attractions['지역코드'] == areaId)]
    print(topAttraction)
    
    topAttractionNames = topAttraction['이름']
    topAttractionLats = topAttraction["위도"]
    topAttractionLngs = topAttraction["경도"]
    topAttractionAreaCodes = topAttraction["지역코드"]
    
    # 위 3가지를 튜플에 담기
    topAttractionPreds = [ Course("관광지",areaCode, index,name, rating,lat, lng) 
                          for index, rating, name, lat, lng, areaCode 
                          in zip(topAttractionIds, topAttractionRatings, topAttractionNames, topAttractionLats, topAttractionLngs,topAttractionAreaCodes)]
    
    return topAttractionPreds


def recommendRestaurants(ai, userId, unVisitedRestaurants, restaurants, areaId, top):
    # AI 객체의 predict()를 이용해서 특정 userId의 평점이 없는 관광지에 대해 평점 예측
    predictions = [ai.predict(str(userId), str(attractionId)) for attractionId in unVisitedRestaurants]
    
    # predictions는 Predction() 하나의 객체르 되어 있기 때문에 예측 평점(est값)을 기준으로 정렬 헤야한다.
    # est값을 반환하는 함수 정의
    def sortedkey_est(pred):
        return pred.est
    
    # sortkey_est 함수로 리스트를 정렬하는 sort 함수의 key인자에 넣어주자
    # 리스트 sort는 디폴트 값이 inplace=True인 것처럼 정렬되어 나온다. reverse=True가 내림차순
    predictions.sort(key=sortedkey_est, reverse=True)
    
    # 상위 n개의 예측값들만 할당
    topPredictions = predictions[:top]
    
    # topPredictions에서 itemId, rating, 명소 이름 뽑아내기
    topRestaurantIds = [int(pred.iid) for pred in topPredictions]
    topRestaurantRatings = [pred.est for pred in topPredictions]
    topRestaurant = restaurants[(restaurants.ID.isin(topRestaurantIds)) & (restaurants['지역코드'] == areaId)]

    topRestaurantNames = topRestaurant['이름']
    topRestaurantLats = topRestaurant["위도"]
    topRestaurantLngs = topRestaurant["경도"]
    topRestaurantAreaCodes = topRestaurant["지역코드"]
    
    # 위 3가지를 튜플에 담기
    topRestaurantPreds = [ Course("식당",areaCode, index,name, rating,lat, lng) 
                          for index, rating, name, lat, lng, areaCode 
                          in zip(topRestaurantIds, topRestaurantRatings, topRestaurantNames, topRestaurantLats, topRestaurantLngs, topRestaurantAreaCodes)]
    
    return topRestaurantPreds


def recommendHotels(ai, userId, unVisitedHotels, hotels, areaId, top = 10):
    # AI 객체의 predict()를 이용해서 특정 userId의 평점이 없는 관광지에 대해 평점 예측
    predictions = [ai.predict(str(userId), str(attractionId)) for attractionId in unVisitedHotels]
    
    # predictions는 Predction() 하나의 객체르 되어 있기 때문에 예측 평점(est값)을 기준으로 정렬 헤야한다.
    # est값을 반환하는 함수 정의
    def sortedkey_est(pred):
        return pred.est
    
    # sortkey_est 함수로 리스트를 정렬하는 sort 함수의 key인자에 넣어주자
    # 리스트 sort는 디폴트 값이 inplace=True인 것처럼 정렬되어 나온다. reverse=True가 내림차순
    predictions.sort(key=sortedkey_est, reverse=True)
    
    # 상위 n개의 예측값들만 할당
    topPredictions = predictions[:top]
    
    # topPredictions에서 itemId, rating, 명소 이름 뽑아내기
    topHotelIds = [int(pred.iid) for pred in topPredictions]
    topHotelRatings = [pred.est for pred in topPredictions]
    topHotel = hotels[(hotels.ID.isin(topHotelIds)) & (hotels['지역코드'] == areaId)]
    topHotelNames = topHotel['이름']
    topHotelLats = topHotel["위도"]
    topHotelLngs = topHotel["경도"]
    topHotelAreaCodes = topHotel["지역코드"]
    # 위 3가지를 튜플에 담기
    topHotelPreds = [ Course("호텔",areaCode, index,name, rating,lat, lng) 
                     for index, rating, name, lat, lng , areaCode
                     in zip(topHotelIds, topHotelRatings, topHotelNames, topHotelLats, topHotelLngs, topHotelAreaCodes)]
    
    return topHotelPreds