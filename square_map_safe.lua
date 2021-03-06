local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local api = {}

function api:start(episode, seed)
  make_map.seedRng(seed)
  api._count = 0
end

function api:commandLine(oldCommandLine)
  return make_map.commandLine(oldCommandLine)
end

function api:createPickup(className)
  return pickups.defaults[className]
end

function api:nextMap()
  map = "*********\n*P LAL A*\n*A ALA  *\n*  *** L*\n*L ALA  *\n*  LAL A*\n*L ***  *\n*  LAL L*\n*A ALA  *\n*********"   
  return make_map.makeMap("square_map", map)
end

return api

