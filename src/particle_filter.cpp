/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"
#include "map.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // init Gaussian distribution
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (auto i = 0; i < num_particles; i++) {
    // init particles
    Particle particle = {
      i,
      dist_x(gen),
      dist_y(gen),
      dist_theta(gen),
      initial_weight
    };
    particles.push_back(particle);

    // init weights
    weights.push_back(initial_weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Make distributions for adding noise
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  for (auto& particle: particles) {
    // if yaw_rate is close to zero use different equation
    if (abs(yaw_rate) > 0.0001) {

      // calculate some points in advance
      const double theta_change = yaw_rate * delta_t;
      const double theta_post = particle.theta + theta_change;
      const double coeff = velocity/yaw_rate;

      // Add measurements to particles
      particle.x += coeff * (sin(theta_post) - sin(particle.theta));
      particle.y += coeff * (cos(particle.theta) - cos(theta_post));
      particle.theta += theta_change;
    }
    else {
      const double coeff = velocity * delta_t;

      // Add measurements to particles
      particle.x += coeff * cos(particle.theta);
      particle.y += coeff * sin(particle.theta);
    }

    // Add noise to the particles
    particle.x += noise_x(gen);
    particle.y += noise_y(gen);
    particle.theta += noise_theta(gen);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    std::vector<LandmarkObs>& car_observations, Map& map_landmarks) {
  // pre-calculate some values to minimise calculation complexity of multi-variate gaussian
  // probability formula inside the loop
  double denominator_x = 2 * pow(std_landmark[0], 2);
  double denominator_y = 2 * pow(std_landmark[1], 2);
  double normalizer = (2 * M_PI * std_landmark[0] * std_landmark[1]);

  for (auto i = 0; i < num_particles; i++) {
    // alias for current particle
    Particle& particle = particles[i];

    // predict observations which should be observed by the car in the particle
    vector<LandmarkObs> reduced_landmarks = reduceLandmarks(particle, sensor_range, map_landmarks);

    // convert car observations to map coordinates
    std::vector<LandmarkObs> map_observations = convertCarToMapCoordinates(particle, car_observations);

    // perform dataAssociation for the predictions and transformed observations on current particle
    dataAssociation(reduced_landmarks, map_observations);

    // re-init weight to update it later
    particle.weight = initial_weight;

    for (auto const& obs: map_observations) {

      // check if we found related map landmark
      if (obs.id < 0) {
        continue;
      }

      LandmarkObs& landmark = reduced_landmarks[obs.id];

      // run final part of multi-variate gaussian probability formula
      double part_x = pow(landmark.x - obs.x, 2) / denominator_x;
      double part_y = pow(landmark.y - obs.y, 2) / denominator_y;
      double exponent = exp(-(part_x + part_y));

      particle.weight *= exponent/normalizer;
    }

    // update weights vector as well
    weights[i] = particle.weight;
  }
}

vector<LandmarkObs> ParticleFilter::reduceLandmarks(Particle& particle, double sensor_range, Map& map_landmarks) {
  // create a vector to hold the map landmark locations predicted to be within sensor range of the particle
  vector<LandmarkObs> reduced;
  LandmarkObs landmark;

  // for each map landmark...
  for (auto const& _landmark: map_landmarks.landmark_list) {

    // it would be faster to init it after the check but in this case code looks ugly...
    landmark = {
      _landmark.id_i,
      _landmark.x_f,
      _landmark.y_f
    };

    // check if we need to get this landmark point into consideration for this particle
    if (dist(landmark.x, landmark.y, particle.x, particle.y) > sensor_range) {
      continue;
    }

    reduced.push_back(landmark);
  }

  return reduced;
}

vector<LandmarkObs> ParticleFilter::convertCarToMapCoordinates(Particle& particle, std::vector<LandmarkObs>& car_observations) {
  // pre-calcuate some values for observations transformation
  const double _sin = sin(particle.theta);
  const double _cos = cos(particle.theta);

  // transform observations to map coordinates system for each particle
  // it would be faster to not use data association method and run associations right here.
  vector<LandmarkObs> map_observations;
  LandmarkObs _obs;

  for (auto const& obs: car_observations) {

    _obs = {
      obs.id,
      _cos*obs.x - _sin*obs.y + particle.x,
      _sin*obs.x + _cos*obs.y + particle.y
    };

    map_observations.push_back(_obs);
  }

  return map_observations;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // define variables
  int target_id;
  double distance, target_distance;

  for (auto& obs: observations) {

    // init variables to keep final closest prediction
    target_distance = numeric_limits<double>::max();

    // init target is a -1 in case we will not find anything
    target_id = -1;

    for (auto j = 0; j < predicted.size(); j++) {

      // set alias for current predicted
      const LandmarkObs& pred = predicted[j];

      // get distance between current/predicted landmarks
      distance = dist(obs.x, obs.y, pred.x, pred.y);

      // in case distance is greater then target distance - continue to next check
      if (target_distance < distance) {
        continue;
      }

      // in other case - set target distance and from current predicted
      target_distance = distance;

      // set index from landmarks list as target id. It will be easier to associate it later in updateWeights procedure
      target_id = j;
    }

    // set the observation's id to the nearest predicted landmark's id
    obs.id = target_id;
  }
}

void ParticleFilter::resample() {
  // new list of particles
  std::vector<Particle> new_particles;

  std::discrete_distribution<int> dist(weights.begin(), weights.end());

  // run resampling process
  for (auto const& particle: particles) {
    new_particles.push_back(std::move(particles[dist(gen)]));
  }

  particles = std::move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
