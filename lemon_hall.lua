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
  map = "*********\n*       *\n*ALLALAL*\n*      P*\n*ALALALA*\n*       *\n*********"
  return make_map.makeMap("hallway_map", map)
end

return api

