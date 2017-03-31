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
  map = "*****************\n*P I  *       *A*\n****  *       * *\n*             *H*\n*               *\n*  *     ****   *\n*  *            *\n****    *H*     *\n*       * *   * *\n*       *A*   * *\n*H**  *** * * * *\n* A*      *** * *\n****          * *\n*             * *\n****       **H* *\n*A I       *A *X*\n*****************"
  return make_map.makeMap("new_map", map)
end

return api
