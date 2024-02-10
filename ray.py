import numpy as np
import matplotlib.pyplot as plt


class Sphere:
  def __init__(self, center=(0.0, 0.0, 0.0), radius=1.0, ambient=(0.1, 0, 0), diffuse=(1, 1, 1), specular=(1, 1, 1), shininess=100, reflection=0.5):
      self.center = np.array(list(center))
      self.radius = radius 
      self.ambient = np.array(list(ambient))
      self.diffuse = np.array(list(diffuse))
      self.specular = np.array(list(specular))
      self.shininess = shininess
      self.reflection = reflection

  def __str__(self): pass 

  @staticmethod
  def run_tests(): pass 


class Light:
  def __init__(self, position=(0.0, 0.0, 0.0), ambient=(1, 1, 1), diffuse=(1, 1, 1), specular=(1, 1, 1)):
      self.position = np.array(list(position))
      self.ambient = np.array(list(ambient))
      self.diffuse = np.array(list(diffuse))
      self.specular = np.array(list(specular))

  def __str__(self): pass 

  @staticmethod
  def run_tests(): pass 


def normalize(vector): return vector / np.linalg.norm(vector)


def reflected(vector, axis): return vector - 2 * np.dot(vector, axis) * axis


def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None


def nearest_intersected_object(objects, ray_origin, ray_direction):
    li = [1, 2, 3, 4]
    li2 = []
    for i in range(1, 5): 
        li2.append(i) 
    li3 = [x for x in range(1, 5)]
    li4 = [x * x for x in range(1, 5)]   #  square of numbers from 1 to 4
    li5 = [(x, x * x) for x in range(1, 5)]   # tuple of (x, x * x) for x from 1 to 4

    distances = [sphere_intersect(obj.center, obj.radius, ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


def ray_trace(width=300, height=200):
  max_depth = 3

  camera = np.array([0, 0, 1])
  ratio = float(width) / height
  screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

  light2 = Light(position=(5, 5, 5))


  objects2 = [
      Sphere(center=( -1.2, 0, -1), radius=0.2, diffuse=(0.7, 0.7, 0)),
      Sphere(center=( -0.8, 0, -1), radius=0.2, diffuse=(0.7, 0.7, 0), shininess=50, reflection=0.8),
      Sphere(center=( -0.4, 0, -1), radius=0.2, diffuse=(0.7, 0.7, 0)),

      Sphere(center=( 0, 0, -1), radius=0.2, diffuse=(0, 0.7, 0)),

      Sphere(center=( 0.4, 0, -1), radius=0.2, diffuse=(0, 0, 0.7)),
      Sphere(center=( 0.8, 0, -1), radius=0.2, diffuse=(0, 0, 0.7), shininess=50, reflection=0.8),
      Sphere(center=(1.2, 0, -1), radius=0.2, diffuse=(0, 0, 0.7)),

      Sphere(center=(0, 1.2, -1), radius=0.2, diffuse=(0.7, 0, 0)),
      Sphere(center=(0, 0.8, -1), radius=0.2, diffuse=(0.7, 0, 0), shininess=50, reflection=0.8),
      Sphere(center=(0, 0.4, -1), radius=0.2, diffuse=(0.7, 0, 0)),

      Sphere(center=(.6, -0.4, -0.8), radius=0.15, diffuse=(0.7, 0, 0), shininess=100, reflection=0.8),
      Sphere(center=(0.8, -0.4, -0.6), radius=0.15, diffuse=(0.7, 0.7, 0), shininess=100, reflection=0.8),
      Sphere(center=(1.0, -0.4, -0.4), radius=0.15, diffuse=(0.7, 0.7, 0.7), shininess=100, reflection=0.8),


      Sphere(center=(9, -9000, 0), radius = 9000 - 0.7, ambient=(0.1, 0.1, 0.1))
  ]

#   objects = [
#       { 'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },

#       { 'center': np.array([  1.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
#       { 'center': np.array([ -1.6, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0.7, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
#       { 'center': np.array([  1.2, 0, -2.4]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0, 0.7, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
#       { 'center': np.array([  1.2, 0,  0.4]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },

#       { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
#       { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
#       { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
#   ]

  image = np.zeros((height, width, 3))
  print(f'{height} lines... ', end=' ', flush=True)

  for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
      for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
          # screen is on origin
          pixel = np.array([x, y, 0])
          origin = camera
          direction = normalize(pixel - origin)

          color = np.zeros((3))
          reflection = 1

          for k in range(max_depth):
              # check for intersections
              nearest_object, min_distance = nearest_intersected_object(objects2, origin, direction)
              if nearest_object is None:
                  break

              intersection = origin + min_distance * direction
              normal_to_surface = normalize(intersection - nearest_object.center)
              shifted_point = intersection + 1e-5 * normal_to_surface
              intersection_to_light = normalize(light2.position - shifted_point)

              _, min_distance = nearest_intersected_object(objects2, shifted_point, intersection_to_light)
              intersection_to_light_distance = np.linalg.norm(light2.position - intersection)
              is_shadowed = min_distance < intersection_to_light_distance

              if is_shadowed:
                  break

              illumination = np.zeros((3))

              # ambient       nearest_object.ambient    * light2.ambient
              illumination += nearest_object.ambient * light2.ambient
              
              #diffuse
              illumination += nearest_object.diffuse * light2.diffuse * np.dot(intersection_to_light, normal_to_surface)

              # specular
              intersection_to_camera = normalize(camera - intersection)
              H = normalize(intersection_to_light + intersection_to_camera)
              illumination += nearest_object.specular * light2.specular * np.dot(normal_to_surface, H) ** (nearest_object.shininess / 4)

              # reflection
              color += reflection * illumination
              reflection *= nearest_object.reflection

              origin = shifted_point
              direction = reflected(direction, normal_to_surface)

          image[i, j] = np.clip(color, 0, 1)
      if (i + 1) % (height / 10) == 0:
        print(i + 1, end=' ', flush=True)

  # plt.show(image)
  plt.imsave('image.png', image)


def main():
    ray_trace(height=400)


if __name__ == '__main__':
    main()
