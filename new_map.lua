local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local api = {}

function api:start(episode, seed)
  make_map.seedRng(seed)
  api._count = -1
end

function api:commandLine(oldCommandLine)
  return make_map.commandLine(oldCommandLine)
end

function api:createPickup(className)
  return pickups.defaults[className]
end

function api:nextMap()
  maps = {"****************\n*P I  *      * *\n**** A*  *  A* *\n*        *   *H*\n*  *   ****    *\n*  *           *\n****   *H*     *\n*    A * *   * *\n*      * *   * *\n*H**  ** ***A* *\n*  *    A    * *\n****      **H* *\n*  I      *  *X*\n****************", "****************\n*P I  *      * *\n****  *  *   * *\n*       A*   *H*\n*  *   ****    *\n*AA*           *\n****   *H*     *\n*      * *   * *\n*      * *A  * *\n*H**  ** *** * *\n*  *         * *\n****      **H* *\n*  I      *A *X*\n****************", "****************\n*P I  *      * *\n****  *  *   * *\n*       A*   *H*\n*  *   ****    *\n*  *           *\n****   *H*    A*\n*      * *   * *\n*A    A* *   * *\n*H**  ** *** * *\n*  *         * *\n****      **H* *\n*A I      *  *X*\n****************", "****************\n*P I  *  A   * *\n****  *  *   *A*\n*        *   *H*\n*  *   ****    *\n* A*           *\n****   *H*     *\n*      * *   * *\n*      * *   * *\n*H**  **A*** * *\n*  *         * *\n****      **H* *\n*A I      *  *X*\n****************", "****************\n*P I  *      *A*\n****  *  *   * *\n*        *   *H*\n*  *   ****    *\n*  *           *\n****   *H*     *\n*      * *   * *\n*      *A*   * *\n*H**  ** *** * *\n* A*         * *\n****      **H* *\n*A I      *A *X*\n****************"}
  api._count = api._count + 1
  return make_map.makeMap("new_map", maps[api._count])
end

return api
