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
  maps = {"***************\n*P I  *     * *\n**** A*  * A*H*\n*             *\n****   *H*    *\n*    A * *  * *\n*H**  ** **A* *\n*  *   A  *H* *\n*  I      I *X*\n***************", "***************\n*P I  *     * *\n****  * A*  *H*\n*AA           *\n****   *H*    *\n*      * *A * *\n*H**  ** ** * *\n*  *      *H* *\n*  I      IA*X*\n***************", "***************\n*P I  *     * *\n****  * A*  *H*\n*             *\n****   *H*   A*\n*A    A* *  * *\n*H**  ** ** * *\n*  *      *H* *\n*A I      I *X*\n***************", "***************\n*P I  *  A  *A*\n****  *  *  *H*\n* A           *\n****   *H*    *\n*      * *  * *\n*H**  **A** * *\n*  *      *H* *\n*A I      I *X*\n***************", "***************\n*P I  *     *A*\n****  *  *  *H*\n*             *\n****   *H*    *\n*      *A*  * *\n*H**  ** ** * *\n* A*      *H* *\n*A I      IA*X*\n***************"}
  api._count = api._count + 1  
  map = maps[api._count]
  if api._count == 5 then
	api._count = 0
  end    
  return make_map.makeMap("new_map", maps[api._count])
end

return api
