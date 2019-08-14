const POINTS = [
  {
    position: new AMap.LngLat(119.430803, 32.147206),
    title: '九华山路金润大道',
  },
  {
    position: new AMap.LngLat(119.433925, 32.152075),
    title: '九华山路五州山路',
  },
  {
    position: new AMap.LngLat(119.433502, 32.158689),
    title: '九华山路白龙山路',
  },
  {
    position: new AMap.LngLat(119.428958, 32.167043),
    title: '九华山路凤凰山路',
  },
  { position: new AMap.LngLat(119.42739, 32.17136), title: '九华山路龙脉路' },
  {
    position: new AMap.LngLat(119.42016, 32.170322),
    title: '檀山路龙脉路',
  },
  { position: new AMap.LngLat(119.420536, 32.187225), title: '檀山路南山路' },
  { position: new AMap.LngLat(119.429087, 32.203059), title: '檀山路中山西路' },
];

// 请补全此处的道路定义， ROADS是记录道路的数组。数组中每个元素是长度为2的，代表路段开始结束点的数组。
const ROADS = [
  { road: [POINTS[0], POINTS[1]], value: 0 },
  { road: [POINTS[1], POINTS[2]], value: 0 },
  { road: [POINTS[2], POINTS[3]], value: 0 },
  { road: [POINTS[3], POINTS[4]], value: 0 },
  { road: [POINTS[6], POINTS[5]], value: 0 },
  { road: [POINTS[7], POINTS[6]], value: 0 },
];

const MAP = new AMap.Map('container', {
  zoom: 11, //级别
  center: POINTS[0].position, //中心点坐标
});

MAP.on('click', (e) => {
  console.log(e.lnglat.lng, e.lnglat.lat);
});

function addPoint(p) {
  MAP.add(new AMap.Marker({ ...p }));
}

function addAllPoints() {
  POINTS.forEach(addPoint);
}

function searchRoad(road_name, callback) {
  AMap.plugin(['AMap.RoadInfoSearch'], function() {
    var roadSearch = new AMap.RoadInfoSearch({
      //构造地点查询类
      pageSize: 5,
      pageIndex: 1,
      city: '镇江',
    });
    //关键字查询
    roadSearch.roadInfoSearchByRoadName(road_name, function(status, result) {
      if (result.roadInfo.length !== 1) {
        console.error('Search result:', result);
        throw Error(`More than one result found for road name ${road_name}`);
      }
      callback(result.roadInfo[0].path);
    });
  });
}

function searchDriving(origin, destination, callback) {
  AMap.plugin('AMap.Driving', function() {
    var drivingSearch = new AMap.Driving();
    drivingSearch.search(origin.position, destination.position, {}, function(
      status,
      result,
    ) {
      const path = Array.prototype.concat(
        ...result.routes[0].steps.map((s) => s.path),
      );
      callback([path]);
    });
  });
}

function drawRedPaths(paths) {
  drawPaths(paths, 'red');
}

function drawPaths(paths, color) {
  paths.forEach((path) => {
    var polyline = new AMap.Polyline({
      path,
      borderWeight: 2, // 线条宽度，默认为 1
      strokeColor: color, // 线条颜色
      lineJoin: 'round', // 折线拐点连接处样式
    });
    MAP.add(polyline);
  });
}

function drawDrivingPaths(origin, destination) {
  addPoint(origin);
  addPoint(destination);
  searchDriving(origin, destination, drawRedPaths);
}

function drawDrivingColorPaths(origin, destination, color) {
  addPoint(origin);
  addPoint(destination);
  searchDriving(origin, destination, (paths) => drawPaths(paths, color));
}

function color(value) {
  if (value == 1) {
    return 'rgb(3, 105, 3)'
  }
  if (value == 2) {
    return 'rgb(4, 204, 30)'
  }
  if (value == 3) {
    return 'rgb(255, 255, 0)'
  }
  if (value == 4) {
    return 'rgb(255, 187, 0)'
  }
  if (value == 5) {
    return 'rgb(202, 124, 5)'
  }
  if (value == 6) {
    return 'red'
  }
  return 'rgba(0,0,0,0)'
}

ROADS.forEach((r) =>
  drawDrivingColorPaths(r.road[0], r.road[1], color(r.value)),
);
